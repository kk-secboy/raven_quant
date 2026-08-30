"""Joint sealed-OOS preregistration for the three public control strategies.

The short, swing and long final-OOS windows necessarily overlap.  Treating
each public control as an unrelated standalone experiment would let whichever
backtest happens to start first prevent the other two from ever running.  This
module instead freezes the complete three-member family before any final OOS
is opened.  Every member still has a one-shot vintage; the joint binding only
allows the three predeclared, different-horizon windows to coexist.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import insert, or_, select, text

from quant_data.database import (
    audit_events,
    backtest_runs,
    jobs,
    oos_vintages,
    open_database,
    row_dict,
    strategies,
    strategy_versions,
    transparent_baseline_pre_result_repairs,
)
from quant_data.history_bounds import GOVERNED_DAILY_STOCK_SCOPE_VERSION
from quant_platform.eligibility import ELIGIBILITY_CONTRACT_VERSION
from quant_platform.strategy_recipes import TRANSPARENT_RESEARCH_BASELINE_IDS
from quant_platform.transparent_baseline_runner import (
    CANONICAL_LF_TARGET_RECIPE_VERSION,
    CANONICAL_LF_TARGET_RUNNER_SHA256,
    DISCRETE_MAX_POSITION_REPAIR_TARGET_RECIPE_VERSION,
    DISCRETE_MAX_POSITION_REPAIR_TARGET_RUNNER_SHA256,
    DISCRETE_MAX_POSITION_REPAIR_TARGET_RUNTIME_BUNDLE_SHA256,
    FAIL_CLOSED_EXECUTION_TARGET_RECIPE_VERSION,
    FAIL_CLOSED_EXECUTION_TARGET_RUNNER_SHA256,
    FAIL_CLOSED_EXECUTION_TARGET_RUNTIME_BUNDLE_SHA256,
    OPTIMIZER_APPLICABILITY_TARGET_RECIPE_VERSION,
    OPTIMIZER_APPLICABILITY_TARGET_RUNNER_SHA256,
    RUNTIME_ALIGNMENT_SOURCE_RUNTIME_BUNDLE_SHA256,
    RUNTIME_ALIGNMENT_TARGET_RUNTIME_BUNDLE_SHA256,
    RUNTIME_INPUT_SCOPE_TARGET_RUNTIME_BUNDLE_SHA256,
    SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RECIPE_VERSION,
    SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RUNNER_SHA256,
    SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RUNTIME_BUNDLE_SHA256,
    TOPK_INDUSTRY_CAPACITY_REPAIR_TARGET_RECIPE_VERSION,
    TOPK_INDUSTRY_CAPACITY_REPAIR_TARGET_RUNNER_SHA256,
    TOPK_INDUSTRY_CAPACITY_REPAIR_TARGET_RUNTIME_BUNDLE_SHA256,
    TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD,
    TRANSPARENT_BASELINE_RUNNER_FIELD,
    TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD,
    TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD,
    target_runner_for_recipe,
    target_runtime_bundle_for_recipe,
)
from quant_platform.transparent_baseline_runner import (
    RUNTIME_ALIGNMENT_TARGET_RECIPE_VERSION as GOVERNED_RUNTIME_TARGET_RECIPE_VERSION,
)
from quant_platform.transparent_baseline_runner import (
    RUNTIME_ALIGNMENT_TARGET_RUNNER_SHA256 as GOVERNED_RUNTIME_TARGET_RUNNER_SHA256,
)
from quant_platform.transparent_baseline_runner import (
    RUNTIME_INPUT_SCOPE_TARGET_RECIPE_VERSION as GOVERNED_INPUT_SCOPE_TARGET_RECIPE_VERSION,
)
from quant_platform.transparent_baseline_runner import (
    RUNTIME_INPUT_SCOPE_TARGET_RUNNER_SHA256 as GOVERNED_INPUT_SCOPE_TARGET_RUNNER_SHA256,
)

LOCKBOX_CONTRACT_VERSION = "transparent-baseline-joint-lockbox-v1"
LOCKBOX_CONTRACT_VERSION_V2 = "transparent-baseline-joint-lockbox-v2"
LOCKBOX_CONTRACT_VERSION_V3 = "transparent-baseline-available-horizons-lockbox-v3"
LOCKBOX_LINK_VERSION = "transparent-baseline-joint-lockbox-link-v1"
LOCKBOX_LINK_VERSION_V2 = "transparent-baseline-available-horizons-link-v2"
LOCKBOX_CONFIG_KEY = "transparent_baseline_joint_lockbox"
BOOTSTRAP_CONFIG_KEY = "transparent_baseline_bootstrap"
UNOPENED_HISTORY_SELECTION_CONTRACT_VERSION = (
    "transparent-baseline-unopened-history-selection-v1"
)
UNOPENED_HISTORY_SELECTION_CONTRACT_VERSION_V2 = (
    "transparent-baseline-unopened-history-selection-v2"
)
UNOPENED_HISTORY_SELECTION_CONTRACT_VERSION_V3 = (
    "transparent-baseline-chained-repair-history-selection-v3"
)
UNOPENED_HISTORY_SELECTION_CONTRACT_VERSION_V4 = (
    "transparent-baseline-multi-generation-repair-history-selection-v4"
)
UNOPENED_HISTORY_SELECTION_POLICY = (
    "exclude-current-recipe-at-earliest-prior-opened-final-oos-v1"
)
UNOPENED_HISTORY_SELECTION_POLICY_V2 = (
    "reuse-exact-preregistered-single-member-pre-result-source-v1"
)
UNOPENED_HISTORY_SELECTION_POLICY_V3 = (
    "reuse-exact-preregistered-chained-single-member-pre-result-source-v1"
)
UNOPENED_HISTORY_SELECTION_POLICY_V4 = (
    "reuse-exact-preregistered-multi-generation-single-member-pre-result-source-v1"
)
PRE_RESULT_REPAIR_ACTION = "transparent_baseline_pre_result_repair_registered"
PRE_RESULT_REPAIR_CONTRACT_VERSION_V1 = "transparent-baseline-pre-result-repair-v1"
PRE_RESULT_REPAIR_CONTRACT_VERSION_V2 = "transparent-baseline-pre-result-repair-v2"
PRE_RESULT_REPAIR_CONTRACT_VERSION_V3 = "transparent-baseline-pre-result-repair-v3"
PRE_RESULT_REPAIR_CONTRACT_VERSION_V4 = "transparent-baseline-pre-result-repair-v4"
PRE_RESULT_REPAIR_CONTRACT_VERSION_V5 = "transparent-baseline-pre-result-repair-v5"
PRE_RESULT_REPAIR_CONTRACT_VERSION_V6 = "transparent-baseline-pre-result-repair-v6"
PRE_RESULT_REPAIR_CONTRACT_VERSION_V7 = "transparent-baseline-pre-result-repair-v7"
PRE_RESULT_REPAIR_CONTRACT_VERSION_V8 = "transparent-baseline-pre-result-repair-v8"
# Keep the historical public name pinned to v1.  Existing receipts and callers
# must not silently acquire the wider v2 shape.
PRE_RESULT_REPAIR_CONTRACT_VERSION = PRE_RESULT_REPAIR_CONTRACT_VERSION_V1
PRE_RESULT_REPAIR_REGISTRY_VERSION = "transparent-baseline-pre-result-repair-registry-v1"
OPTIMIZER_APPLICABILITY_REPAIR_GENERATION = "v7-to-v8-optimizer-applicability"
OPTIMIZER_APPLICABILITY_REASON = "topk_equal_weight_optimizer_applicability"
OPTIMIZER_APPLICABILITY_SOURCE_COMMIT = (
    "79a88b3d3e2ed03b1785aa6ff6d578ff1d516860"
)
OPTIMIZER_APPLICABILITY_SOURCE_RECIPE_VERSION = (
    "qlib-rdagent-single-mainline-2026-08-30-v7"
)
OPTIMIZER_APPLICABILITY_SOURCE_BACKTEST_IDS = frozenset(
    {
        "f51d7fa2f4fd463e97fd5f6990b3721c",
        "8090c21aa11546bd9d59f732975afc25",
        "9c8a75ac646f452e8a5666bacd708936",
    }
)
CANONICAL_LF_PACKAGING_REPAIR_GENERATION = "v8-to-v9-canonical-lf-packaging"
CANONICAL_LF_PACKAGING_REASON = "transparent_baseline_runner_canonical_lf_packaging"
CANONICAL_LF_PACKAGING_SOURCE_COMMIT = (
    "413024ff5971115d4eb6a33872ebcafe9619cc9e"
)
CANONICAL_LF_PACKAGING_SOURCE_RECIPE_VERSION = (
    OPTIMIZER_APPLICABILITY_TARGET_RECIPE_VERSION
)
CANONICAL_LF_PACKAGING_TARGET_RECIPE_VERSION = CANONICAL_LF_TARGET_RECIPE_VERSION
CANONICAL_LF_PACKAGING_SOURCE_EXPECTED_RUNNER_SHA256 = (
    OPTIMIZER_APPLICABILITY_TARGET_RUNNER_SHA256
)
CANONICAL_LF_PACKAGING_SOURCE_OBSERVED_RUNNER_SHA256 = (
    "1ce281eb2e0141922215b9966f1ba91e09d073019a2e6e4b6a4614049b769e44"
)
CANONICAL_LF_PACKAGING_TARGET_RUNNER_SHA256 = CANONICAL_LF_TARGET_RUNNER_SHA256
CANONICAL_LF_PACKAGING_CONTRACT_VERSION = "git-archive-canonical-lf-v1"
CANONICAL_LF_PACKAGING_SOURCE_BACKTEST_IDS = frozenset(
    {
        "1ca979f22a0d4e2981e6e5c5e478f583",
        "7b4386b52cf24d6d913dae3b232fbe7f",
        "0c5f5a0b66284a66ba68225afd22b623",
    }
)
CANONICAL_LF_PACKAGING_SOURCE_BINDINGS = {
    "1ca979f22a0d4e2981e6e5c5e478f583": {
        "job_id": "34f6675d3c79426babeaa08b14af6b5f",
        "strategy_version_id": "e919845154444f93adceed9b14057289",
    },
    "7b4386b52cf24d6d913dae3b232fbe7f": {
        "job_id": "e698fe80cfdf432d9ead1e810c409cdd",
        "strategy_version_id": "d3b6b4918ef14350805c138290540731",
    },
    "0c5f5a0b66284a66ba68225afd22b623": {
        "job_id": "3107865f05c547fb9e5d631e2b34a135",
        "strategy_version_id": "acdd946a84d34197a68b1f9c33ea3553",
    },
}
RUNTIME_ALIGNMENT_REPAIR_GENERATION = "v9-to-v10-runtime-contract-alignment"
RUNTIME_ALIGNMENT_CONTRACT_VERSION = (
    "transparent-baseline-runtime-contract-alignment-v1"
)
RUNTIME_ALIGNMENT_SOURCE_COMMIT = "3e3bc56a2daf7e35206e94c8d36a4c9ffb2a9fd6"
RUNTIME_ALIGNMENT_SOURCE_RECIPE_VERSION = CANONICAL_LF_TARGET_RECIPE_VERSION
RUNTIME_ALIGNMENT_TARGET_RECIPE_VERSION = GOVERNED_RUNTIME_TARGET_RECIPE_VERSION
RUNTIME_ALIGNMENT_SOURCE_RUNNER_SHA256 = CANONICAL_LF_TARGET_RUNNER_SHA256
RUNTIME_ALIGNMENT_TARGET_RUNNER_SHA256 = GOVERNED_RUNTIME_TARGET_RUNNER_SHA256
RUNTIME_ALIGNMENT_SOURCE_BUNDLE_SHA256 = (
    RUNTIME_ALIGNMENT_SOURCE_RUNTIME_BUNDLE_SHA256
)
RUNTIME_ALIGNMENT_TARGET_BUNDLE_SHA256 = (
    RUNTIME_ALIGNMENT_TARGET_RUNTIME_BUNDLE_SHA256
)
RUNTIME_ALIGNMENT_BENCHMARK_REASON = "compiled-market-benchmark-close-binding"
RUNTIME_ALIGNMENT_INDUSTRY_REASON = "reporting-benchmark-relative-cap-leak"
RUNTIME_ALIGNMENT_BENCHMARK_ERROR = (
    "ValueError: market-trend rule has incomplete benchmark close history"
)
RUNTIME_ALIGNMENT_INDUSTRY_ERROR = (
    "ValueError: industry constraints leave too few eligible instruments"
)
RUNTIME_ALIGNMENT_SOURCE_DATASET = "cn-20080101-20260828-v6-79a88b3"
RUNTIME_ALIGNMENT_SOURCE_BINDINGS = {
    "ef927367c58143448ebeaab2eceab5f1": {
        "job_id": "86d178fcdbaa4081bc3126810e87f0ef",
        "strategy_version_id": "28ef148affcb4ce1ba516ea733500887",
        "reason_code": RUNTIME_ALIGNMENT_INDUSTRY_REASON,
        "error": RUNTIME_ALIGNMENT_INDUSTRY_ERROR,
        "periods": {
            "historical_start": "2009-01-08",
            "historical_end": "2021-06-18",
            "start": "2022-07-06",
            "end": "2025-08-14",
        },
        "artifact_inventory_sha256": (
            "76f50fbb3c3a5bd6ebedae14d1ac70e621b6a2d85b1f3e3594071eaac93bf2c7"
        ),
    },
    "f795119754ea41c2960a710e2626bd19": {
        "job_id": "4c8693f79bec47f5afe5560f376983ee",
        "strategy_version_id": "7c52079068274f2e9e2e1dcbffce9559",
        "reason_code": RUNTIME_ALIGNMENT_BENCHMARK_REASON,
        "error": RUNTIME_ALIGNMENT_BENCHMARK_ERROR,
        "periods": {
            "historical_start": "2014-02-11",
            "historical_end": "2025-07-10",
            "start": "2025-08-08",
            "end": "2026-08-21",
        },
        "artifact_inventory_sha256": (
            "a5086dafde910b7f70fb29952a086b9ee9163186021a28697de1f9455cb1599c"
        ),
    },
    "e05b237e364c4811a97ff5e2b49fc68c": {
        "job_id": "e2b16680f3fc4dc0b1c1037ddb759e9e",
        "strategy_version_id": "3399bd080d6c418d9ec972721344c656",
        "reason_code": RUNTIME_ALIGNMENT_BENCHMARK_REASON,
        "error": RUNTIME_ALIGNMENT_BENCHMARK_ERROR,
        "periods": {
            "historical_start": "2011-08-10",
            "historical_end": "2023-07-17",
            "start": "2024-01-22",
            "end": "2026-02-26",
        },
        "artifact_inventory_sha256": (
            "0716b0e590c3662c34c09f06f4d6aebf163a4e06be80a494a5031ca970c1fa16"
        ),
    },
}
RUNTIME_ALIGNMENT_SOURCE_BACKTEST_IDS = frozenset(
    RUNTIME_ALIGNMENT_SOURCE_BINDINGS
)
RUNTIME_INPUT_SCOPE_REPAIR_GENERATION = "v10-to-v11-runtime-input-scope"
RUNTIME_INPUT_SCOPE_CONTRACT_VERSION = "transparent-baseline-runtime-input-scope-v1"
RUNTIME_INPUT_SCOPE_SOURCE_COMMIT = "bb4d1139f848f0e39b82d13f7c4d58ff7659d8d2"
RUNTIME_INPUT_SCOPE_SOURCE_RECIPE_VERSION = RUNTIME_ALIGNMENT_TARGET_RECIPE_VERSION
RUNTIME_INPUT_SCOPE_TARGET_RECIPE_VERSION = GOVERNED_INPUT_SCOPE_TARGET_RECIPE_VERSION
RUNTIME_INPUT_SCOPE_SOURCE_RUNNER_SHA256 = RUNTIME_ALIGNMENT_TARGET_RUNNER_SHA256
RUNTIME_INPUT_SCOPE_TARGET_RUNNER_SHA256 = GOVERNED_INPUT_SCOPE_TARGET_RUNNER_SHA256
RUNTIME_INPUT_SCOPE_SOURCE_BUNDLE_SHA256 = RUNTIME_ALIGNMENT_TARGET_BUNDLE_SHA256
RUNTIME_INPUT_SCOPE_TARGET_BUNDLE_SHA256 = (
    RUNTIME_INPUT_SCOPE_TARGET_RUNTIME_BUNDLE_SHA256
)
RUNTIME_INPUT_SCOPE_TOPK_BENCHMARK_REASON = (
    "topk-unused-benchmark-industry-metadata-leak"
)
RUNTIME_INPUT_SCOPE_TREND_REASON = "trend-break-nonholding-history-scope-leak"
RUNTIME_INPUT_SCOPE_VALUATION_REASON = (
    "valuation-entry-missing-exposure-global-failure"
)
# These codes describe target-runtime semantics that are materially different
# from v10 but were not the observed error marker of any source backtest.  Keep
# them separate from ``reason_codes`` so the repair receipt never rewrites the
# historical failure evidence.
RUNTIME_INPUT_SCOPE_TARGET_CHANGE_CODES = (
    "missing-5d-extension-evidence-per-instrument-new-entry-rejection",
    "missing-trend-evidence-holding-continuity-new-entry-rejection",
    "shared-governed-style-exposure-snapshot-backtest-recommendation",
    "topk-benchmark-weight-non-consumption",
)
RUNTIME_INPUT_SCOPE_TOPK_BENCHMARK_ERROR = (
    "ValueError: benchmark constituents are missing point-in-time industries"
)
RUNTIME_INPUT_SCOPE_TREND_ERROR = (
    "ValueError: trend-break rule has incomplete close history"
)
RUNTIME_INPUT_SCOPE_VALUATION_ERROR = (
    "ValueError: valuation-regime value exposures are incomplete"
)
RUNTIME_INPUT_SCOPE_SOURCE_DATASET = "cn-20080101-20260828-v6-79a88b3"
RUNTIME_INPUT_SCOPE_SOURCE_DATASET_IDENTITY_SHA256 = (
    "4771fc24680dcdca18fbcf73c887f604aad70d5f586c25116102f75d51d6e042"
)
RUNTIME_INPUT_SCOPE_SOURCE_DATASET_LINEAGE_ID = (
    "6e169632f9f322db93856f0e7ade3c9436b310e3509806f68d0b47db837e1b6d"
)
RUNTIME_INPUT_SCOPE_SOURCE_BATCH_SHA256 = (
    "9bc6a3017c2c718497dd53395da43c6d7392ceefdd0a268587c48d77813bcd65"
)
RUNTIME_INPUT_SCOPE_SOURCE_ARTIFACT_INVENTORIES_SHA256 = (
    "c2727672b33fed580841721551c45df1a11e864dd899cb6d473c220821da0dc0"
)
RUNTIME_INPUT_SCOPE_SOURCE_BINDINGS = {
    "00f1b9171d3d4ec3a04ade9ddec7d061": {
        "job_id": "36c20231ba7e48f285f4c3e16fc30ef6",
        "strategy_version_id": "a9b9dd6a34d34ea983c4754912a87820",
        "reason_code": RUNTIME_INPUT_SCOPE_TOPK_BENCHMARK_REASON,
        "error": RUNTIME_INPUT_SCOPE_TOPK_BENCHMARK_ERROR,
        "periods": {
            "historical_start": "2014-02-11",
            "historical_end": "2025-07-10",
            "start": "2025-08-08",
            "end": "2026-08-21",
        },
        "artifact_inventory_sha256": (
            "d401693394d966fdb28bff4cede757dec225a26468ee8da43c2329ba7ce5d58e"
        ),
    },
    "4aa9972029d34793848f2c5a3e4ddb43": {
        "job_id": "88cd85f298c649a1b97130a996a8011e",
        "strategy_version_id": "59b5315df5364438a60c57ee0d5a997f",
        "reason_code": RUNTIME_INPUT_SCOPE_TREND_REASON,
        "error": RUNTIME_INPUT_SCOPE_TREND_ERROR,
        "periods": {
            "historical_start": "2011-08-10",
            "historical_end": "2023-07-17",
            "start": "2024-01-22",
            "end": "2026-02-26",
        },
        "artifact_inventory_sha256": (
            "39af316bfeb9e2c3141d435f23354cbbd0575f981293da1fd627d84136ca610e"
        ),
    },
    "04e84f759929477a9cae3de4ba1749fe": {
        "job_id": "78c111bfb10f467b8e197fb7dde68f0e",
        "strategy_version_id": "7b9476ece2b2455c8f61b36c46beddea",
        "reason_code": RUNTIME_INPUT_SCOPE_VALUATION_REASON,
        "error": RUNTIME_INPUT_SCOPE_VALUATION_ERROR,
        "periods": {
            "historical_start": "2009-01-08",
            "historical_end": "2021-06-18",
            "start": "2022-07-06",
            "end": "2025-08-14",
        },
        "artifact_inventory_sha256": (
            "412710b28ede1acaa3d27210fd6cbfeded8bf157e731b51eefd2a30c3d11c89e"
        ),
    },
}
RUNTIME_INPUT_SCOPE_SOURCE_BACKTEST_IDS = frozenset(
    RUNTIME_INPUT_SCOPE_SOURCE_BINDINGS
)

FILL_AWARE_HOLDING_AGE_REPAIR_GENERATION = (
    "v13-to-v15-fill-aware-holding-age"
)
FILL_AWARE_HOLDING_AGE_RUNTIME_CONTRACT_VERSION = (
    "transparent-baseline-fill-aware-holding-age-repair-v1"
)
FILL_AWARE_HOLDING_AGE_REASON = (
    "holding-age-reconciliation-after-unfilled-or-partial-exit"
)
FILL_AWARE_HOLDING_AGE_ERROR = (
    "ValueError: holding-period policy requires complete holding-age state"
)
FILL_AWARE_HOLDING_AGE_TARGET_CHANGE_CODES = (
    "preserve-and-reconcile-holding-age-from-actual-holdings-after-unfilled-or-partial-exit",
)
FILL_AWARE_HOLDING_AGE_SOURCE_COMMIT = (
    "ed5c8b3d0ea118dddbec7b1b5cf3c82b8cd72c08"
)
FILL_AWARE_HOLDING_AGE_SOURCE_RECIPE_VERSION = (
    FAIL_CLOSED_EXECUTION_TARGET_RECIPE_VERSION
)
FILL_AWARE_HOLDING_AGE_TARGET_RECIPE_VERSION = (
    SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RECIPE_VERSION
)
FILL_AWARE_HOLDING_AGE_SOURCE_RUNNER_SHA256 = (
    FAIL_CLOSED_EXECUTION_TARGET_RUNNER_SHA256
)
FILL_AWARE_HOLDING_AGE_TARGET_RUNNER_SHA256 = (
    SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RUNNER_SHA256
)
FILL_AWARE_HOLDING_AGE_SOURCE_BUNDLE_SHA256 = (
    FAIL_CLOSED_EXECUTION_TARGET_RUNTIME_BUNDLE_SHA256
)
FILL_AWARE_HOLDING_AGE_TARGET_BUNDLE_SHA256 = (
    SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RUNTIME_BUNDLE_SHA256
)
FILL_AWARE_HOLDING_AGE_SOURCE_BATCH_SHA256 = (
    "34e96967cdb021913489ec95eebfd82eda68df7bdbee92ead837e467c3e7af96"
)
FILL_AWARE_HOLDING_AGE_SOURCE_DATASET = (
    "cn-20080101-20260828-v7-failclosed-ed5c8b3"
)
FILL_AWARE_HOLDING_AGE_SOURCE_DATASET_IDENTITY_SHA256 = (
    "eab69dff43abcc77e60f47ad51d2182c7bfd5d90f5d36c479e3de61cebf768f2"
)
FILL_AWARE_HOLDING_AGE_SOURCE_DATASET_LINEAGE_ID = (
    "1b97efcc3956b4be2dfebff7567efd4707c73d806f9d0cb2b63540d9dedd852e"
)
FILL_AWARE_HOLDING_AGE_SOURCE_ARTIFACT_INVENTORIES_SHA256 = (
    "7a83e76cde3fbc8a62583da41aef00613d4e15479a71b6e83a03b069d6005e01"
)
FILL_AWARE_HOLDING_AGE_RECEIPT_SHA256 = (
    "4cbd548de4cc38df791e4abeed59423c333769ffbc55f055291d457d4d332aeb"
)
FILL_AWARE_HOLDING_AGE_SOURCE_SELECTION_SHA256 = (
    "145161dbb19e8448995dcaaf781a9d433cfb0ce99ff9812367d5b1fbf1860076"
)
FILL_AWARE_HOLDING_AGE_SOURCE_UNAVAILABLE_HORIZONS_SHA256 = (
    "b9f36421f0047ab024eec0f2b20909a105c2a57e17dc94a50aa25e632f8c3644"
)
FILL_AWARE_HOLDING_AGE_UNAVAILABLE_EVIDENCE_SHA256S = frozenset(
    {
        "b57d3dba4ab3d2284b2a001cb30a7b2862a8659c3738d5fc23419e7e7a6b5729",
        "62f5642a4ad024cc24d23ec34b9ce760a552f560c783612b63dab459ba00d14d",
    }
)
FILL_AWARE_HOLDING_AGE_SOURCE_BINDINGS = {
    "51959d99d199410fa13a718658d03773": {
        "job_id": "4f4c1f51e74e4ed18cb6cb9f6ed757c7",
        "strategy_version_id": "520b4c76f75445a88039a6f02d0d7dd1",
        "reason_code": FILL_AWARE_HOLDING_AGE_REASON,
        "error": FILL_AWARE_HOLDING_AGE_ERROR,
        "periods": {
            "historical_start": "2008-01-02",
            "historical_end": "2018-10-10",
            "start": "2018-11-08",
            "end": "2019-11-20",
        },
        "artifact_inventory_sha256": (
            "883add2be6f985fead0528b6d8dd2faa8e418a47041af71901a0d677ab5ff381"
        ),
    }
}
FILL_AWARE_HOLDING_AGE_SOURCE_BACKTEST_IDS = frozenset(
    FILL_AWARE_HOLDING_AGE_SOURCE_BINDINGS
)

DISCRETE_MAX_POSITION_REPAIR_GENERATION = (
    "v15-to-v16-discrete-max-position-weight"
)
DISCRETE_MAX_POSITION_RUNTIME_CONTRACT_VERSION = (
    "transparent-baseline-discrete-max-position-repair-v1"
)
DISCRETE_MAX_POSITION_REASON = "post-discretization-max-position-weight-hard-cap"
DISCRETE_MAX_POSITION_ERROR = (
    "ValueError: post-discretization hard constraint violation: "
    "max_position_weight[account]"
)
DISCRETE_MAX_POSITION_TARGET_CHANGE_CODES = (
    "reduce-tradable-position-to-largest-whole-lot-within-hard-cap",
    "preserve-hard-position-cap-after-turnover-scaling",
    "override-soft-hold-rules-for-tradable-inherited-overweight-risk-reduction",
)
DISCRETE_MAX_POSITION_SOURCE_COMMIT = (
    "efec9ceca53b5f62e986e38b3d701c8dda7c8f56"
)
DISCRETE_MAX_POSITION_SOURCE_RECIPE_VERSION = (
    SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RECIPE_VERSION
)
DISCRETE_MAX_POSITION_TARGET_RECIPE_VERSION = (
    DISCRETE_MAX_POSITION_REPAIR_TARGET_RECIPE_VERSION
)
DISCRETE_MAX_POSITION_SOURCE_RUNNER_SHA256 = (
    SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RUNNER_SHA256
)
DISCRETE_MAX_POSITION_TARGET_RUNNER_SHA256 = (
    DISCRETE_MAX_POSITION_REPAIR_TARGET_RUNNER_SHA256
)
DISCRETE_MAX_POSITION_SOURCE_BUNDLE_SHA256 = (
    SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RUNTIME_BUNDLE_SHA256
)
DISCRETE_MAX_POSITION_TARGET_BUNDLE_SHA256 = (
    DISCRETE_MAX_POSITION_REPAIR_TARGET_RUNTIME_BUNDLE_SHA256
)
DISCRETE_MAX_POSITION_SOURCE_BATCH_SHA256 = (
    "913ebb7aa7dcdb1267d31ee1079eb9206068e25bb8a1f858e0a85b7d37dac157"
)
DISCRETE_MAX_POSITION_SOURCE_DATASET = (
    "cn-20080101-20260828-v7-failclosed-ed5c8b3"
)
DISCRETE_MAX_POSITION_SOURCE_DATASET_IDENTITY_SHA256 = (
    "eab69dff43abcc77e60f47ad51d2182c7bfd5d90f5d36c479e3de61cebf768f2"
)
DISCRETE_MAX_POSITION_SOURCE_DATASET_LINEAGE_ID = (
    "1b97efcc3956b4be2dfebff7567efd4707c73d806f9d0cb2b63540d9dedd852e"
)
DISCRETE_MAX_POSITION_SOURCE_ARTIFACT_INVENTORIES_SHA256 = (
    "eaa39ec9858a37d4d71754832e2fd5a71ecf55230f426ecf725f3d188979fe51"
)
DISCRETE_MAX_POSITION_SOURCE_SELECTION_SHA256 = (
    "81431777c685845c00e12f49bc4c901d130c4704922216af3e407623b775727b"
)
DISCRETE_MAX_POSITION_SOURCE_UNAVAILABLE_HORIZONS_SHA256 = (
    "b9f36421f0047ab024eec0f2b20909a105c2a57e17dc94a50aa25e632f8c3644"
)
DISCRETE_MAX_POSITION_UNAVAILABLE_EVIDENCE_SHA256S = frozenset(
    {
        "b57d3dba4ab3d2284b2a001cb30a7b2862a8659c3738d5fc23419e7e7a6b5729",
        "62f5642a4ad024cc24d23ec34b9ce760a552f560c783612b63dab459ba00d14d",
    }
)
DISCRETE_MAX_POSITION_SOURCE_BINDINGS = {
    "afa2257f2f6f4f0fb70cd285be5a9605": {
        "job_id": "4c1927be70314e7cb60e1e7992fa5d51",
        "strategy_version_id": "9c9009051df64c2c91b879cb99421c68",
        "reason_code": DISCRETE_MAX_POSITION_REASON,
        "error": DISCRETE_MAX_POSITION_ERROR,
        "periods": {
            "historical_start": "2008-01-02",
            "historical_end": "2018-10-10",
            "start": "2018-11-08",
            "end": "2019-11-20",
        },
        "artifact_inventory_sha256": (
            "2349dea933a4f15f2d8a322ad60d6dd4ca1a8d1fa2ea99edbca3532c560b27c9"
        ),
    }
}
DISCRETE_MAX_POSITION_SOURCE_BACKTEST_IDS = frozenset(
    DISCRETE_MAX_POSITION_SOURCE_BINDINGS
)
DISCRETE_MAX_POSITION_RECEIPT_SHA256 = (
    "dbe47a338ee6fd75c5b6dc471775aaa0155697fa7c31012561f362c7cfb5128a"
)

TOPK_INDUSTRY_CAPACITY_REPAIR_GENERATION = (
    "v16-to-v17-topk-industry-capacity-partial-cash"
)
TOPK_INDUSTRY_CAPACITY_RUNTIME_CONTRACT_VERSION = (
    "transparent-baseline-topk-industry-capacity-partial-cash-repair-v1"
)
TOPK_INDUSTRY_CAPACITY_REASON = "topk-industry-capacity-partial-cash"
TOPK_INDUSTRY_CAPACITY_ERROR = (
    "ValueError: industry constraints leave too few eligible instruments"
)
TOPK_INDUSTRY_CAPACITY_TARGET_CHANGE_CODES = (
    "topk_industry_capacity_partial_cash",
)
TOPK_INDUSTRY_CAPACITY_SOURCE_COMMIT = (
    "92c722b89450d9b25549ac777ca171fd163c57ca"
)
TOPK_INDUSTRY_CAPACITY_SOURCE_RECIPE_VERSION = (
    DISCRETE_MAX_POSITION_REPAIR_TARGET_RECIPE_VERSION
)
TOPK_INDUSTRY_CAPACITY_TARGET_RECIPE_VERSION = (
    TOPK_INDUSTRY_CAPACITY_REPAIR_TARGET_RECIPE_VERSION
)
TOPK_INDUSTRY_CAPACITY_SOURCE_RUNNER_SHA256 = (
    DISCRETE_MAX_POSITION_REPAIR_TARGET_RUNNER_SHA256
)
TOPK_INDUSTRY_CAPACITY_TARGET_RUNNER_SHA256 = (
    TOPK_INDUSTRY_CAPACITY_REPAIR_TARGET_RUNNER_SHA256
)
TOPK_INDUSTRY_CAPACITY_SOURCE_BUNDLE_SHA256 = (
    DISCRETE_MAX_POSITION_REPAIR_TARGET_RUNTIME_BUNDLE_SHA256
)
TOPK_INDUSTRY_CAPACITY_TARGET_BUNDLE_SHA256 = (
    TOPK_INDUSTRY_CAPACITY_REPAIR_TARGET_RUNTIME_BUNDLE_SHA256
)
TOPK_INDUSTRY_CAPACITY_SOURCE_BATCH_SHA256 = (
    "9721d472a343295294d6199de9c17bfcd932f1494b88e9dfa18928a3c682f5c9"
)
TOPK_INDUSTRY_CAPACITY_SOURCE_DATASET = (
    "cn-20080101-20260828-v7-failclosed-ed5c8b3"
)
TOPK_INDUSTRY_CAPACITY_SOURCE_DATASET_IDENTITY_SHA256 = (
    "eab69dff43abcc77e60f47ad51d2182c7bfd5d90f5d36c479e3de61cebf768f2"
)
TOPK_INDUSTRY_CAPACITY_SOURCE_DATASET_LINEAGE_ID = (
    "1b97efcc3956b4be2dfebff7567efd4707c73d806f9d0cb2b63540d9dedd852e"
)
TOPK_INDUSTRY_CAPACITY_SOURCE_ARTIFACT_INVENTORIES_SHA256 = (
    "fb4a4a2e457ae0e674ba638af7c0e11c604a9847375d1470bdf4e3d255630435"
)
TOPK_INDUSTRY_CAPACITY_SOURCE_SELECTION_SHA256 = (
    "b86ed4a518a6d58b049ad3aeb31e058daaff75de67619b316f39f8fb2f2b1d67"
)
TOPK_INDUSTRY_CAPACITY_SOURCE_UNAVAILABLE_HORIZONS_SHA256 = (
    "b9f36421f0047ab024eec0f2b20909a105c2a57e17dc94a50aa25e632f8c3644"
)
TOPK_INDUSTRY_CAPACITY_UNAVAILABLE_EVIDENCE_SHA256S = frozenset(
    {
        "b57d3dba4ab3d2284b2a001cb30a7b2862a8659c3738d5fc23419e7e7a6b5729",
        "62f5642a4ad024cc24d23ec34b9ce760a552f560c783612b63dab459ba00d14d",
    }
)
TOPK_INDUSTRY_CAPACITY_SOURCE_BINDINGS = {
    "b1a0f92acf3145ba9a6d887d459d1129": {
        "job_id": "5587b686787f47dca72abad634d18ec3",
        "strategy_version_id": "934a776a9d6c4c538411473165d3c93e",
        "reason_code": TOPK_INDUSTRY_CAPACITY_REASON,
        "error": TOPK_INDUSTRY_CAPACITY_ERROR,
        "periods": {
            "historical_start": "2008-01-02",
            "historical_end": "2018-10-10",
            "start": "2018-11-08",
            "end": "2019-11-20",
        },
        "artifact_inventory_sha256": (
            "14f0a3e2262095c4229254ac52ad7a9a0d6ed5f28ce3a6ee78fb61981d58837e"
        ),
    }
}
TOPK_INDUSTRY_CAPACITY_SOURCE_BACKTEST_IDS = frozenset(
    TOPK_INDUSTRY_CAPACITY_SOURCE_BINDINGS
)

_PRE_RESULT_REPAIR_REASON_CODES = frozenset(
    {
        "empty_eligible_session_pandas_concat_failure",
        "pre_2023_bse_history_outside_governed_scope",
    }
)
_PRE_RESULT_FAILURES = {
    "empty_eligible_session_pandas_concat_failure": (
        "cannot concatenate unaligned mixed dimensional NDFrame objects"
    ),
    "pre_2023_bse_history_outside_governed_scope": (
        "formal execution starts before native price-limit controls are complete"
    ),
}
OPTIMIZER_APPLICABILITY_FAILURE = (
    "optimizer requires 60 complete point-in-time return observations"
)
OPTIMIZER_APPLICABILITY_ERROR = f"ValueError: {OPTIMIZER_APPLICABILITY_FAILURE}"
CANONICAL_LF_PACKAGING_ERROR = (
    "transparent v8 runner bytes differ from the repair authorization"
)

_RECIPE_HORIZONS = {
    "short_relative_strength": "short_1_5d",
    "swing_trend": "swing_1_6m",
    "long_quality_value": "long_1_3y",
}
_MEMBER_KEYS = {
    "recipe_id",
    "recipe_version",
    "recipe_sha256",
    "horizon_profile",
    "base_config_sha256",
    "baseline_definition_sha256",
    "research_window_contract_sha256",
    "historical_start",
    "historical_end",
    "test_start",
    "test_end",
}
_RUNNER_BOUND_MEMBER_KEYS = _MEMBER_KEYS | {TRANSPARENT_BASELINE_RUNNER_FIELD}
_RUNTIME_BOUND_MEMBER_KEYS = _RUNNER_BOUND_MEMBER_KEYS | {
    TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD
}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return normalized


def _require_identifier(value: Any, *, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 32 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field} must be a lowercase 32-character identifier")
    return normalized


def _require_worker_image_digest(value: Any, *, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if (
        not normalized.startswith("sha256:")
        or len(normalized) != 71
        or any(character not in "0123456789abcdef" for character in normalized[7:])
    ):
        raise ValueError(f"{field} must be a sha256 image digest")
    return normalized


def _ordered_calendar(calendar_days: Sequence[Any]) -> list[str]:
    try:
        ordered = [date.fromisoformat(str(day)).isoformat() for day in calendar_days]
    except (TypeError, ValueError) as exc:
        raise ValueError("transparent baseline selection calendar is invalid") from exc
    if not ordered or ordered != sorted(set(ordered)):
        raise ValueError(
            "transparent baseline selection calendar must be ordered and unique"
        )
    return ordered


def _normalize_prior_batch(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("transparent baseline prior-batch evidence is invalid")
    expected_keys = {
        "batch_sha256",
        "recipe_version",
        "earliest_final_oos_start",
        "latest_final_oos_end",
        "members",
        "members_sha256",
    }
    if set(value) != expected_keys:
        raise ValueError("transparent baseline prior-batch evidence is invalid")
    batch_sha256 = _require_sha256(
        value.get("batch_sha256"), field="prior batch_sha256"
    )
    recipe_version = str(value.get("recipe_version") or "").strip()
    if not recipe_version:
        raise ValueError("transparent baseline prior recipe version is required")
    raw_members = value.get("members")
    if not isinstance(raw_members, Sequence) or isinstance(raw_members, (str, bytes)):
        raise ValueError("transparent baseline prior-batch members are invalid")
    members: list[dict[str, Any]] = []
    member_keys = {
        "oos_vintage_id",
        "strategy_version_id",
        "recipe_id",
        "horizon_profile",
        "recipe_version",
        "test_start",
        "test_end",
        "first_opened_at",
        "sealed_member_set_sha256",
    }
    for raw in raw_members:
        if not isinstance(raw, Mapping) or set(raw) != member_keys:
            raise ValueError("transparent baseline prior-batch member is invalid")
        recipe_id = str(raw.get("recipe_id") or "").strip()
        if recipe_id not in _RECIPE_HORIZONS:
            raise ValueError("transparent baseline prior-batch recipe is invalid")
        horizon = str(raw.get("horizon_profile") or "").strip()
        if horizon != _RECIPE_HORIZONS[recipe_id]:
            raise ValueError("transparent baseline prior-batch horizon changed")
        if str(raw.get("recipe_version") or "").strip() != recipe_version:
            raise ValueError("transparent baseline prior batch mixes recipe versions")
        try:
            test_start = date.fromisoformat(str(raw.get("test_start")))
            test_end = date.fromisoformat(str(raw.get("test_end")))
            first_opened_at = datetime.fromisoformat(
                str(raw.get("first_opened_at"))
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "transparent baseline prior-batch dates are invalid"
            ) from exc
        if test_start > test_end:
            raise ValueError("transparent baseline prior-batch OOS is not ordered")
        if first_opened_at.tzinfo is None:
            raise ValueError(
                "transparent baseline prior-batch opening time must be timezone-aware"
            )
        members.append(
            {
                "oos_vintage_id": _require_identifier(
                    raw.get("oos_vintage_id"), field="oos_vintage_id"
                ),
                "strategy_version_id": _require_identifier(
                    raw.get("strategy_version_id"), field="strategy_version_id"
                ),
                "recipe_id": recipe_id,
                "horizon_profile": horizon,
                "recipe_version": recipe_version,
                "test_start": test_start.isoformat(),
                "test_end": test_end.isoformat(),
                "first_opened_at": first_opened_at.isoformat(),
                "sealed_member_set_sha256": _require_sha256(
                    raw.get("sealed_member_set_sha256"),
                    field="sealed_member_set_sha256",
                ),
            }
        )
    members.sort(key=lambda item: item["recipe_id"])
    member_recipe_ids = [str(item["recipe_id"]) for item in members]
    if (
        not member_recipe_ids
        or len(member_recipe_ids) != len(set(member_recipe_ids))
        or not set(member_recipe_ids) <= set(TRANSPARENT_RESEARCH_BASELINE_IDS)
    ):
        raise ValueError(
            "transparent baseline prior batch must contain one to three unique horizons"
        )
    if canonical_sha256(members) != _require_sha256(
        value.get("members_sha256"), field="prior members_sha256"
    ):
        raise ValueError("transparent baseline prior-batch member digest changed")
    earliest = min(item["test_start"] for item in members)
    latest = max(item["test_end"] for item in members)
    if (
        str(value.get("earliest_final_oos_start") or "") != earliest
        or str(value.get("latest_final_oos_end") or "") != latest
    ):
        raise ValueError("transparent baseline prior-batch OOS bounds changed")
    return {
        "batch_sha256": batch_sha256,
        "recipe_version": recipe_version,
        "earliest_final_oos_start": earliest,
        "latest_final_oos_end": latest,
        "members": members,
        "members_sha256": canonical_sha256(members),
    }


def build_unopened_history_selection(
    *,
    calendar_days: Sequence[Any],
    current_recipe_version: str,
    prior_batches: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze the latest calendar prefix never opened by a prior recipe.

    This function consumes only immutable OOS-window identities. It never
    reads prior results, metrics or repair receipts.
    """

    calendar = _ordered_calendar(calendar_days)
    recipe_version = str(current_recipe_version or "").strip()
    if not recipe_version:
        raise ValueError("transparent baseline current recipe version is required")
    normalized_batches = sorted(
        (_normalize_prior_batch(value) for value in prior_batches),
        key=lambda item: item["batch_sha256"],
    )
    if len({item["batch_sha256"] for item in normalized_batches}) != len(
        normalized_batches
    ):
        raise ValueError("transparent baseline prior-batch evidence is duplicated")
    if any(
        item["recipe_version"] == recipe_version for item in normalized_batches
    ):
        raise ValueError("current transparent recipe cannot move its own history cutoff")
    earliest_prior_start = (
        min(item["earliest_final_oos_start"] for item in normalized_batches)
        if normalized_batches
        else None
    )
    if earliest_prior_start is None:
        selected_calendar = calendar
        cutoff_rule = "latest_real_trading_session"
        selection_mode = "latest_history_no_prior_transparent_batch"
    else:
        selected_calendar = [day for day in calendar if day < earliest_prior_start]
        if not selected_calendar:
            raise ValueError(
                "transparent baseline history has no trading session before the prior OOS"
            )
        cutoff_rule = "real_trading_session_preceding_earliest_prior_final_oos"
        selection_mode = "unopened_history_before_prior_transparent_oos"
    prior_batches_sha256 = canonical_sha256(normalized_batches)
    payload = {
        "contract_version": UNOPENED_HISTORY_SELECTION_CONTRACT_VERSION,
        "selection_policy": UNOPENED_HISTORY_SELECTION_POLICY,
        "selection_mode": selection_mode,
        "cutoff_rule": cutoff_rule,
        "current_recipe_version": recipe_version,
        "source_calendar_start": calendar[0],
        "source_calendar_end": calendar[-1],
        "source_calendar_trading_days": len(calendar),
        "source_calendar_sha256": canonical_sha256(calendar),
        "earliest_prior_final_oos_start": earliest_prior_start,
        "selected_calendar_end": selected_calendar[-1],
        "selected_calendar_trading_days": len(selected_calendar),
        "selected_calendar_sha256": canonical_sha256(selected_calendar),
        "prior_batches": normalized_batches,
        "prior_batches_sha256": prior_batches_sha256,
        "prior_results_or_metrics_read": False,
        "prior_windows_treatment": "ordinary_historical_validation_only",
    }
    return {**payload, "selection_sha256": canonical_sha256(payload)}


def _single_member_repair_profile_for_target(
    current_recipe_version: str,
) -> dict[str, Any]:
    if current_recipe_version == FILL_AWARE_HOLDING_AGE_TARGET_RECIPE_VERSION:
        return {
            "label": "fill-aware holding-age",
            "contract_version": PRE_RESULT_REPAIR_CONTRACT_VERSION_V6,
            "source_batch_sha256": FILL_AWARE_HOLDING_AGE_SOURCE_BATCH_SHA256,
            "source_recipe_version": FILL_AWARE_HOLDING_AGE_SOURCE_RECIPE_VERSION,
            "target_recipe_version": FILL_AWARE_HOLDING_AGE_TARGET_RECIPE_VERSION,
            "source_selection_sha256": FILL_AWARE_HOLDING_AGE_SOURCE_SELECTION_SHA256,
            "source_bindings": FILL_AWARE_HOLDING_AGE_SOURCE_BINDINGS,
            "source_dataset": FILL_AWARE_HOLDING_AGE_SOURCE_DATASET,
            "source_dataset_identity_sha256": (
                FILL_AWARE_HOLDING_AGE_SOURCE_DATASET_IDENTITY_SHA256
            ),
            "source_dataset_lineage_id": (
                FILL_AWARE_HOLDING_AGE_SOURCE_DATASET_LINEAGE_ID
            ),
            "source_runner_sha256": FILL_AWARE_HOLDING_AGE_SOURCE_RUNNER_SHA256,
            "target_runner_sha256": FILL_AWARE_HOLDING_AGE_TARGET_RUNNER_SHA256,
            "source_bundle_sha256": FILL_AWARE_HOLDING_AGE_SOURCE_BUNDLE_SHA256,
            "target_bundle_sha256": FILL_AWARE_HOLDING_AGE_TARGET_BUNDLE_SHA256,
            "source_unavailable_horizons_sha256": (
                FILL_AWARE_HOLDING_AGE_SOURCE_UNAVAILABLE_HORIZONS_SHA256
            ),
            "unavailable_evidence_sha256s": (
                FILL_AWARE_HOLDING_AGE_UNAVAILABLE_EVIDENCE_SHA256S
            ),
            "source_history_contract": UNOPENED_HISTORY_SELECTION_CONTRACT_VERSION,
            "target_history_contract": UNOPENED_HISTORY_SELECTION_CONTRACT_VERSION_V2,
            "target_selection_policy": UNOPENED_HISTORY_SELECTION_POLICY_V2,
            "target_selection_mode": (
                "exact_preregistered_single_member_pre_result_repair"
            ),
            "source_receipt_sha256": None,
        }
    if current_recipe_version == DISCRETE_MAX_POSITION_TARGET_RECIPE_VERSION:
        return {
            "label": "discrete max-position",
            "contract_version": PRE_RESULT_REPAIR_CONTRACT_VERSION_V7,
            "source_batch_sha256": DISCRETE_MAX_POSITION_SOURCE_BATCH_SHA256,
            "source_recipe_version": DISCRETE_MAX_POSITION_SOURCE_RECIPE_VERSION,
            "target_recipe_version": DISCRETE_MAX_POSITION_TARGET_RECIPE_VERSION,
            "source_selection_sha256": DISCRETE_MAX_POSITION_SOURCE_SELECTION_SHA256,
            "source_bindings": DISCRETE_MAX_POSITION_SOURCE_BINDINGS,
            "source_dataset": DISCRETE_MAX_POSITION_SOURCE_DATASET,
            "source_dataset_identity_sha256": (
                DISCRETE_MAX_POSITION_SOURCE_DATASET_IDENTITY_SHA256
            ),
            "source_dataset_lineage_id": (
                DISCRETE_MAX_POSITION_SOURCE_DATASET_LINEAGE_ID
            ),
            "source_runner_sha256": DISCRETE_MAX_POSITION_SOURCE_RUNNER_SHA256,
            "target_runner_sha256": DISCRETE_MAX_POSITION_TARGET_RUNNER_SHA256,
            "source_bundle_sha256": DISCRETE_MAX_POSITION_SOURCE_BUNDLE_SHA256,
            "target_bundle_sha256": DISCRETE_MAX_POSITION_TARGET_BUNDLE_SHA256,
            "source_unavailable_horizons_sha256": (
                DISCRETE_MAX_POSITION_SOURCE_UNAVAILABLE_HORIZONS_SHA256
            ),
            "unavailable_evidence_sha256s": (
                DISCRETE_MAX_POSITION_UNAVAILABLE_EVIDENCE_SHA256S
            ),
            "source_history_contract": (
                UNOPENED_HISTORY_SELECTION_CONTRACT_VERSION_V2
            ),
            "target_history_contract": (
                UNOPENED_HISTORY_SELECTION_CONTRACT_VERSION_V3
            ),
            "target_selection_policy": UNOPENED_HISTORY_SELECTION_POLICY_V3,
            "target_selection_mode": (
                "exact_preregistered_chained_single_member_pre_result_repair"
            ),
            "source_receipt_sha256": FILL_AWARE_HOLDING_AGE_RECEIPT_SHA256,
        }
    if current_recipe_version == TOPK_INDUSTRY_CAPACITY_TARGET_RECIPE_VERSION:
        return {
            "label": "topk industry-capacity",
            "contract_version": PRE_RESULT_REPAIR_CONTRACT_VERSION_V8,
            "source_batch_sha256": TOPK_INDUSTRY_CAPACITY_SOURCE_BATCH_SHA256,
            "source_recipe_version": TOPK_INDUSTRY_CAPACITY_SOURCE_RECIPE_VERSION,
            "target_recipe_version": TOPK_INDUSTRY_CAPACITY_TARGET_RECIPE_VERSION,
            "source_selection_sha256": TOPK_INDUSTRY_CAPACITY_SOURCE_SELECTION_SHA256,
            "source_bindings": TOPK_INDUSTRY_CAPACITY_SOURCE_BINDINGS,
            "source_dataset": TOPK_INDUSTRY_CAPACITY_SOURCE_DATASET,
            "source_dataset_identity_sha256": (
                TOPK_INDUSTRY_CAPACITY_SOURCE_DATASET_IDENTITY_SHA256
            ),
            "source_dataset_lineage_id": (
                TOPK_INDUSTRY_CAPACITY_SOURCE_DATASET_LINEAGE_ID
            ),
            "source_runner_sha256": TOPK_INDUSTRY_CAPACITY_SOURCE_RUNNER_SHA256,
            "target_runner_sha256": TOPK_INDUSTRY_CAPACITY_TARGET_RUNNER_SHA256,
            "source_bundle_sha256": TOPK_INDUSTRY_CAPACITY_SOURCE_BUNDLE_SHA256,
            "target_bundle_sha256": TOPK_INDUSTRY_CAPACITY_TARGET_BUNDLE_SHA256,
            "source_unavailable_horizons_sha256": (
                TOPK_INDUSTRY_CAPACITY_SOURCE_UNAVAILABLE_HORIZONS_SHA256
            ),
            "unavailable_evidence_sha256s": (
                TOPK_INDUSTRY_CAPACITY_UNAVAILABLE_EVIDENCE_SHA256S
            ),
            "source_history_contract": (
                UNOPENED_HISTORY_SELECTION_CONTRACT_VERSION_V3
            ),
            "target_history_contract": (
                UNOPENED_HISTORY_SELECTION_CONTRACT_VERSION_V4
            ),
            "target_selection_policy": UNOPENED_HISTORY_SELECTION_POLICY_V4,
            "target_selection_mode": (
                "exact_preregistered_multi_generation_single_member_pre_result_repair"
            ),
            "source_receipt_sha256": DISCRETE_MAX_POSITION_RECEIPT_SHA256,
        }
    raise ValueError("single-member repair target recipe is not allowlisted")


def _validate_single_member_repair_source_batch(
    value: Any,
    *,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    source_batch = _normalize_prior_batch(value)
    members = list(source_batch["members"])
    source_binding = next(iter(dict(profile["source_bindings"]).values()))
    if (
        source_batch["batch_sha256"] != profile["source_batch_sha256"]
        or source_batch["recipe_version"] != profile["source_recipe_version"]
        or len(members) != 1
        or members[0]["recipe_id"] != "short_relative_strength"
        or members[0]["horizon_profile"] != "short_1_5d"
        or members[0]["recipe_version"] != profile["source_recipe_version"]
        or members[0]["strategy_version_id"]
        != source_binding["strategy_version_id"]
        or members[0]["test_start"] != source_binding["periods"]["start"]
        or members[0]["test_end"] != source_binding["periods"]["end"]
    ):
        raise ValueError(f"{profile['label']} repair source batch changed")
    return source_batch


def _validate_fill_aware_repair_source_batch(value: Any) -> dict[str, Any]:
    return _validate_single_member_repair_source_batch(
        value,
        profile=_single_member_repair_profile_for_target(
            FILL_AWARE_HOLDING_AGE_TARGET_RECIPE_VERSION
        ),
    )


def build_pre_result_repair_history_selection(
    *,
    calendar_days: Sequence[Any],
    current_recipe_version: str,
    source_selection: Mapping[str, Any],
    repaired_source_batch: Mapping[str, Any],
    repair_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebind one exact preregistered short OOS to its governed successor.

    The repaired v13 batch is deliberately *not* added to ``prior_batches``:
    doing so would move the final OOS to an earlier window.  Instead this v2
    evidence names that one excluded batch and its pre-result receipt while
    preserving every ordinary prior batch from the frozen v13 selection.
    """

    calendar = _ordered_calendar(calendar_days)
    current = str(current_recipe_version or "").strip()
    profile = _single_member_repair_profile_for_target(current)
    source = validate_unopened_history_selection(
        source_selection,
        calendar_days=calendar,
    )
    if (
        source["contract_version"] != profile["source_history_contract"]
        or source["current_recipe_version"] != profile["source_recipe_version"]
        or source["selection_sha256"] != profile["source_selection_sha256"]
        or source["selected_calendar_end"] != "2019-11-27"
        or source["selected_calendar_trading_days"] != 2897
    ):
        raise ValueError(f"{profile['label']} source history selection changed")
    source_batch = _validate_single_member_repair_source_batch(
        repaired_source_batch,
        profile=profile,
    )
    receipt = validate_pre_result_repair_receipt(repair_receipt)
    if (
        receipt["contract_version"] != profile["contract_version"]
        or receipt["source_batch_sha256"] != profile["source_batch_sha256"]
    ):
        raise ValueError(f"{profile['label']} history receipt changed")
    ordinary = build_unopened_history_selection(
        calendar_days=calendar,
        current_recipe_version=current,
        prior_batches=list(source["prior_batches"]),
    )
    for field in (
        "source_calendar_start",
        "source_calendar_end",
        "source_calendar_trading_days",
        "source_calendar_sha256",
        "earliest_prior_final_oos_start",
        "selected_calendar_end",
        "selected_calendar_trading_days",
        "selected_calendar_sha256",
        "prior_batches_sha256",
    ):
        if ordinary[field] != source[field]:
            raise ValueError(f"{profile['label']} history selection moved")
    payload = {
        **{key: value for key, value in ordinary.items() if key != "selection_sha256"},
        "contract_version": profile["target_history_contract"],
        "selection_policy": profile["target_selection_policy"],
        "selection_mode": profile["target_selection_mode"],
        "prior_windows_treatment": (
            "ordinary_historical_validation_plus_exact_pre_result_source_exclusion"
        ),
        "repaired_source_batch": source_batch,
        "repaired_source_batch_sha256": profile["source_batch_sha256"],
        "source_history_selection_sha256": source["selection_sha256"],
        "repair_receipt_sha256": receipt["receipt_sha256"],
        "performance_information_used": False,
    }
    if profile["source_receipt_sha256"] is not None:
        source_receipt_sha256 = _require_sha256(
            source.get("repair_receipt_sha256"),
            field="source_repair_receipt_sha256",
        )
        if source_receipt_sha256 != profile["source_receipt_sha256"]:
            raise ValueError(f"{profile['label']} predecessor receipt changed")
        payload["source_repair_receipt_sha256"] = source_receipt_sha256
    return {**payload, "selection_sha256": canonical_sha256(payload)}


def validate_unopened_history_selection(
    value: Any,
    *,
    calendar_days: Sequence[Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("transparent baseline unopened-history selection is required")
    history_contract = value.get("contract_version")
    if history_contract in {
        UNOPENED_HISTORY_SELECTION_CONTRACT_VERSION_V2,
        UNOPENED_HISTORY_SELECTION_CONTRACT_VERSION_V3,
        UNOPENED_HISTORY_SELECTION_CONTRACT_VERSION_V4,
    }:
        has_predecessor_receipt = history_contract in {
            UNOPENED_HISTORY_SELECTION_CONTRACT_VERSION_V3,
            UNOPENED_HISTORY_SELECTION_CONTRACT_VERSION_V4,
        }
        expected_repair_keys = {
            "contract_version",
            "selection_policy",
            "selection_mode",
            "cutoff_rule",
            "current_recipe_version",
            "source_calendar_start",
            "source_calendar_end",
            "source_calendar_trading_days",
            "source_calendar_sha256",
            "earliest_prior_final_oos_start",
            "selected_calendar_end",
            "selected_calendar_trading_days",
            "selected_calendar_sha256",
            "prior_batches",
            "prior_batches_sha256",
            "prior_results_or_metrics_read",
            "prior_windows_treatment",
            "repaired_source_batch",
            "repaired_source_batch_sha256",
            "source_history_selection_sha256",
            "repair_receipt_sha256",
            "performance_information_used",
            "selection_sha256",
        }
        if has_predecessor_receipt:
            expected_repair_keys.add("source_repair_receipt_sha256")
        if set(value) != expected_repair_keys:
            raise ValueError("transparent baseline repair history selection is invalid")
        raw_batches = value.get("prior_batches")
        if not isinstance(raw_batches, list) or any(
            not isinstance(item, Mapping) for item in raw_batches
        ):
            raise ValueError("transparent baseline prior-batch evidence is invalid")
        normalized_batches = sorted(
            (_normalize_prior_batch(item) for item in raw_batches),
            key=lambda item: item["batch_sha256"],
        )
        profile = _single_member_repair_profile_for_target(
            str(value.get("current_recipe_version") or "")
        )
        if (
            profile["contract_version"]
            not in {
                PRE_RESULT_REPAIR_CONTRACT_VERSION_V6,
                PRE_RESULT_REPAIR_CONTRACT_VERSION_V7,
                PRE_RESULT_REPAIR_CONTRACT_VERSION_V8,
            }
            or profile["target_history_contract"] != history_contract
        ):
            raise ValueError("transparent baseline repair history generation changed")
        source_batch = _validate_single_member_repair_source_batch(
            value.get("repaired_source_batch"),
            profile=profile,
        )
        payload = dict(value)
        selection_sha256 = _require_sha256(
            payload.pop("selection_sha256", None), field="selection_sha256"
        )
        if (
            canonical_sha256(payload) != selection_sha256
            or value.get("selection_policy") != profile["target_selection_policy"]
            or value.get("selection_mode") != profile["target_selection_mode"]
            or value.get("current_recipe_version") != profile["target_recipe_version"]
            or value.get("repaired_source_batch_sha256")
            != profile["source_batch_sha256"]
            or source_batch["batch_sha256"]
            != profile["source_batch_sha256"]
            or _require_sha256(
                value.get("source_history_selection_sha256"),
                field="source_history_selection_sha256",
            )
            != profile["source_selection_sha256"]
            or _require_sha256(
                value.get("repair_receipt_sha256"),
                field="repair_receipt_sha256",
            )
            != value.get("repair_receipt_sha256")
            or value.get("performance_information_used") is not False
            or value.get("prior_results_or_metrics_read") is not False
            or value.get("prior_windows_treatment")
            != "ordinary_historical_validation_plus_exact_pre_result_source_exclusion"
            or value.get("prior_batches_sha256")
            != canonical_sha256(normalized_batches)
            or any(
                item["recipe_version"] == value.get("current_recipe_version")
                for item in normalized_batches
            )
            or value.get("selected_calendar_end") != "2019-11-27"
            or value.get("selected_calendar_trading_days") != 2897
            or (
                has_predecessor_receipt
                and _require_sha256(
                    value.get("source_repair_receipt_sha256"),
                    field="source_repair_receipt_sha256",
                )
                != profile["source_receipt_sha256"]
            )
        ):
            raise ValueError("transparent baseline repair history selection changed")
        if calendar_days is not None:
            ordinary = build_unopened_history_selection(
                calendar_days=calendar_days,
                current_recipe_version=profile["target_recipe_version"],
                prior_batches=normalized_batches,
            )
            for field in (
                "cutoff_rule",
                "source_calendar_start",
                "source_calendar_end",
                "source_calendar_trading_days",
                "source_calendar_sha256",
                "earliest_prior_final_oos_start",
                "selected_calendar_end",
                "selected_calendar_trading_days",
                "selected_calendar_sha256",
            ):
                if value.get(field) != ordinary[field]:
                    raise ValueError(
                        "transparent baseline repair history selection or cutoff changed"
                    )
        return dict(value)
    expected_keys = {
        "contract_version",
        "selection_policy",
        "selection_mode",
        "cutoff_rule",
        "current_recipe_version",
        "source_calendar_start",
        "source_calendar_end",
        "source_calendar_trading_days",
        "source_calendar_sha256",
        "earliest_prior_final_oos_start",
        "selected_calendar_end",
        "selected_calendar_trading_days",
        "selected_calendar_sha256",
        "prior_batches",
        "prior_batches_sha256",
        "prior_results_or_metrics_read",
        "prior_windows_treatment",
        "selection_sha256",
    }
    if set(value) != expected_keys:
        raise ValueError("transparent baseline unopened-history selection is invalid")
    raw_batches = value.get("prior_batches")
    if not isinstance(raw_batches, Sequence) or isinstance(raw_batches, (str, bytes)):
        raise ValueError("transparent baseline prior-batch evidence is invalid")
    if len([item for item in raw_batches if isinstance(item, Mapping)]) != len(
        raw_batches
    ):
        raise ValueError("transparent baseline prior-batch evidence is invalid")
    if calendar_days is None:
        # Without the source calendar, validate the recursively sealed payload
        # directly; callers that select a prefix always provide the calendar.
        normalized_batches = sorted(
            (_normalize_prior_batch(item) for item in raw_batches),
            key=lambda item: item["batch_sha256"],
        )
        payload = dict(value)
        selection_sha256 = _require_sha256(
            payload.pop("selection_sha256", None), field="selection_sha256"
        )
        if canonical_sha256(payload) != selection_sha256:
            raise ValueError("transparent baseline history selection digest changed")
        try:
            source_start = date.fromisoformat(str(value.get("source_calendar_start")))
            source_end = date.fromisoformat(str(value.get("source_calendar_end")))
            selected_end = date.fromisoformat(str(value.get("selected_calendar_end")))
            source_days = int(value.get("source_calendar_trading_days"))
            selected_days = int(value.get("selected_calendar_trading_days"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "transparent baseline history selection calendar is invalid"
            ) from exc
        earliest = (
            min(item["earliest_final_oos_start"] for item in normalized_batches)
            if normalized_batches
            else None
        )
        expected_mode = (
            "unopened_history_before_prior_transparent_oos"
            if normalized_batches
            else "latest_history_no_prior_transparent_batch"
        )
        expected_rule = (
            "real_trading_session_preceding_earliest_prior_final_oos"
            if normalized_batches
            else "latest_real_trading_session"
        )
        if (
            value.get("contract_version")
            != UNOPENED_HISTORY_SELECTION_CONTRACT_VERSION
            or value.get("selection_policy") != UNOPENED_HISTORY_SELECTION_POLICY
            or value.get("selection_mode") != expected_mode
            or value.get("cutoff_rule") != expected_rule
            or not str(value.get("current_recipe_version") or "").strip()
            or source_days < 1
            or selected_days < 1
            or selected_days > source_days
            or not source_start <= selected_end <= source_end
            or value.get("earliest_prior_final_oos_start") != earliest
            or (earliest is not None and selected_end >= date.fromisoformat(earliest))
            or _require_sha256(
                value.get("source_calendar_sha256"),
                field="source_calendar_sha256",
            )
            != value.get("source_calendar_sha256")
            or _require_sha256(
                value.get("selected_calendar_sha256"),
                field="selected_calendar_sha256",
            )
            != value.get("selected_calendar_sha256")
            or value.get("prior_results_or_metrics_read") is not False
            or value.get("prior_windows_treatment")
            != "ordinary_historical_validation_only"
            or value.get("prior_batches_sha256")
            != canonical_sha256(normalized_batches)
            or any(
                item["recipe_version"] == value.get("current_recipe_version")
                for item in normalized_batches
            )
            or (
                not normalized_batches
                and (
                    selected_end != source_end
                    or selected_days != source_days
                    or value.get("selected_calendar_sha256")
                    != value.get("source_calendar_sha256")
                )
            )
        ):
            raise ValueError("transparent baseline history selection policy changed")
        return dict(value)
    normalized = build_unopened_history_selection(
        calendar_days=calendar_days,
        current_recipe_version=str(value.get("current_recipe_version") or ""),
        prior_batches=[item for item in raw_batches if isinstance(item, Mapping)],
    )
    if dict(value) != normalized:
        raise ValueError("transparent baseline history selection or cutoff changed")
    return normalized


def validate_pre_result_repair_receipt(value: Any) -> dict[str, Any]:
    """Validate an explicitly allowlisted no-performance baseline repair.

    This is intentionally narrower than a general retry token. It seals the
    exact prior attempts while they have no result/metrics and names the
    corrected recipe/data contracts before a replacement lockbox is opened.
    V1 remains byte-for-byte compatible with its historical receipt shape;
    V2 authorizes only the v7-to-v8 optimizer-applicability repair; V3
    authorizes only the three exact v8 failures caused before the canonical-LF
    release package could execute the already-frozen runner source; V4 seals
    the exact v9 runtime-contract failures; V5 seals the exact v10 runtime
    input-scope failures and their complete partial-artifact inventories, plus
    a separate exact target-change list, before any corrected code can open a
    fresh OOS scope. V6, V7 and V8 are exact one-member generations: they bind
    the failed v13 holding-age, v15 discrete-position and v16 topk industry-
    capacity attempts, respectively, before the same OOS can be opened by
    their governed successors.
    """

    if not isinstance(value, Mapping):
        raise ValueError("transparent baseline pre-result repair receipt is required")
    common_keys = {
        "contract_version",
        "source_release_commit",
        "target_recipe_version",
        "target_eligibility_contract",
        "target_stock_scope_contract",
        "reason_codes",
        "performance_information_used",
        "members",
        "receipt_sha256",
    }
    contract_version = value.get("contract_version")
    if contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V1:
        keys = common_keys
        expected_reasons = _PRE_RESULT_REPAIR_REASON_CODES
        failure_markers = _PRE_RESULT_FAILURES
        repair_generation: str | None = None
    elif contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V2:
        keys = common_keys | {
            "repair_generation",
            TRANSPARENT_BASELINE_RUNNER_FIELD,
        }
        expected_reasons = frozenset({OPTIMIZER_APPLICABILITY_REASON})
        failure_markers = {
            OPTIMIZER_APPLICABILITY_REASON: OPTIMIZER_APPLICABILITY_FAILURE
        }
        repair_generation = str(value.get("repair_generation") or "").strip()
        if repair_generation != OPTIMIZER_APPLICABILITY_REPAIR_GENERATION:
            raise ValueError("transparent baseline repair generation is not allowlisted")
    elif contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V3:
        keys = common_keys | {
            "repair_generation",
            "source_runner_expected_sha256",
            "source_runner_observed_sha256",
            TRANSPARENT_BASELINE_RUNNER_FIELD,
            "packaging_contract_version",
        }
        expected_reasons = frozenset({CANONICAL_LF_PACKAGING_REASON})
        failure_markers = {
            CANONICAL_LF_PACKAGING_REASON: CANONICAL_LF_PACKAGING_ERROR
        }
        repair_generation = str(value.get("repair_generation") or "").strip()
        if repair_generation != CANONICAL_LF_PACKAGING_REPAIR_GENERATION:
            raise ValueError("transparent baseline repair generation is not allowlisted")
    elif contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V4:
        keys = common_keys | {
            "repair_generation",
            "source_runner_sha256",
            TRANSPARENT_BASELINE_RUNNER_FIELD,
            "source_runtime_bundle_sha256",
            "target_runtime_bundle_sha256",
            "runtime_contract_version",
            "source_artifact_inventories_sha256",
        }
        expected_reasons = frozenset(
            {
                RUNTIME_ALIGNMENT_BENCHMARK_REASON,
                RUNTIME_ALIGNMENT_INDUSTRY_REASON,
            }
        )
        failure_markers = {
            RUNTIME_ALIGNMENT_BENCHMARK_REASON: RUNTIME_ALIGNMENT_BENCHMARK_ERROR,
            RUNTIME_ALIGNMENT_INDUSTRY_REASON: RUNTIME_ALIGNMENT_INDUSTRY_ERROR,
        }
        repair_generation = str(value.get("repair_generation") or "").strip()
        if repair_generation != RUNTIME_ALIGNMENT_REPAIR_GENERATION:
            raise ValueError("transparent baseline repair generation is not allowlisted")
    elif contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V5:
        keys = common_keys | {
            "repair_generation",
            "source_batch_sha256",
            "source_dataset_identity_sha256",
            "source_dataset_lineage_id",
            "source_runner_sha256",
            TRANSPARENT_BASELINE_RUNNER_FIELD,
            "source_runtime_bundle_sha256",
            "target_runtime_bundle_sha256",
            "runtime_contract_version",
            "source_artifact_inventories_sha256",
            "target_change_codes",
        }
        expected_reasons = frozenset(
            {
                RUNTIME_INPUT_SCOPE_TOPK_BENCHMARK_REASON,
                RUNTIME_INPUT_SCOPE_TREND_REASON,
                RUNTIME_INPUT_SCOPE_VALUATION_REASON,
            }
        )
        failure_markers = {
            RUNTIME_INPUT_SCOPE_TOPK_BENCHMARK_REASON: (
                RUNTIME_INPUT_SCOPE_TOPK_BENCHMARK_ERROR
            ),
            RUNTIME_INPUT_SCOPE_TREND_REASON: RUNTIME_INPUT_SCOPE_TREND_ERROR,
            RUNTIME_INPUT_SCOPE_VALUATION_REASON: RUNTIME_INPUT_SCOPE_VALUATION_ERROR,
        }
        repair_generation = str(value.get("repair_generation") or "").strip()
        if repair_generation != RUNTIME_INPUT_SCOPE_REPAIR_GENERATION:
            raise ValueError("transparent baseline repair generation is not allowlisted")
    elif contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V6:
        keys = common_keys | {
            "repair_generation",
            "source_batch_sha256",
            "source_dataset_identity_sha256",
            "source_dataset_lineage_id",
            "source_runner_sha256",
            TRANSPARENT_BASELINE_RUNNER_FIELD,
            "source_runtime_bundle_sha256",
            "target_runtime_bundle_sha256",
            "runtime_contract_version",
            "source_artifact_inventories_sha256",
            "source_unopened_history_selection_sha256",
            "source_unavailable_horizons_sha256",
            "source_unavailable_evidence_sha256s",
            "target_change_codes",
        }
        expected_reasons = frozenset({FILL_AWARE_HOLDING_AGE_REASON})
        failure_markers = {
            FILL_AWARE_HOLDING_AGE_REASON: FILL_AWARE_HOLDING_AGE_ERROR
        }
        repair_generation = str(value.get("repair_generation") or "").strip()
        if repair_generation != FILL_AWARE_HOLDING_AGE_REPAIR_GENERATION:
            raise ValueError("transparent baseline repair generation is not allowlisted")
    elif contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V7:
        keys = common_keys | {
            "repair_generation",
            "source_batch_sha256",
            "source_dataset_identity_sha256",
            "source_dataset_lineage_id",
            "source_runner_sha256",
            TRANSPARENT_BASELINE_RUNNER_FIELD,
            "source_runtime_bundle_sha256",
            "target_runtime_bundle_sha256",
            "runtime_contract_version",
            "source_artifact_inventories_sha256",
            "source_unopened_history_selection_sha256",
            "source_unavailable_horizons_sha256",
            "source_unavailable_evidence_sha256s",
            "target_change_codes",
        }
        expected_reasons = frozenset({DISCRETE_MAX_POSITION_REASON})
        failure_markers = {
            DISCRETE_MAX_POSITION_REASON: DISCRETE_MAX_POSITION_ERROR
        }
        repair_generation = str(value.get("repair_generation") or "").strip()
        if repair_generation != DISCRETE_MAX_POSITION_REPAIR_GENERATION:
            raise ValueError("transparent baseline repair generation is not allowlisted")
    elif contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V8:
        keys = common_keys | {
            "repair_generation",
            "source_batch_sha256",
            "source_dataset_identity_sha256",
            "source_dataset_lineage_id",
            "source_runner_sha256",
            TRANSPARENT_BASELINE_RUNNER_FIELD,
            "source_runtime_bundle_sha256",
            "target_runtime_bundle_sha256",
            "runtime_contract_version",
            "source_artifact_inventories_sha256",
            "source_unopened_history_selection_sha256",
            "source_unavailable_horizons_sha256",
            "source_unavailable_evidence_sha256s",
            "target_change_codes",
        }
        expected_reasons = frozenset({TOPK_INDUSTRY_CAPACITY_REASON})
        failure_markers = {
            TOPK_INDUSTRY_CAPACITY_REASON: TOPK_INDUSTRY_CAPACITY_ERROR
        }
        repair_generation = str(value.get("repair_generation") or "").strip()
        if repair_generation != TOPK_INDUSTRY_CAPACITY_REPAIR_GENERATION:
            raise ValueError("transparent baseline repair generation is not allowlisted")
    else:
        raise ValueError("transparent baseline pre-result repair contract is invalid")
    if set(value) != keys:
        raise ValueError("transparent baseline pre-result repair contract is invalid")
    payload = dict(value)
    receipt_sha256 = _require_sha256(
        payload.pop("receipt_sha256", None), field="receipt_sha256"
    )
    if canonical_sha256(payload) != receipt_sha256:
        raise ValueError("transparent baseline pre-result repair digest changed")
    commit = str(value.get("source_release_commit") or "").strip().lower()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("transparent baseline repair source commit is invalid")
    target_recipe_version = str(value.get("target_recipe_version") or "").strip()
    if not target_recipe_version:
        raise ValueError("transparent baseline repair target recipe is required")
    if contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V2 and (
        commit != OPTIMIZER_APPLICABILITY_SOURCE_COMMIT
        or target_recipe_version != OPTIMIZER_APPLICABILITY_TARGET_RECIPE_VERSION
        or _require_sha256(
            value.get(TRANSPARENT_BASELINE_RUNNER_FIELD),
            field=TRANSPARENT_BASELINE_RUNNER_FIELD,
        )
        != OPTIMIZER_APPLICABILITY_TARGET_RUNNER_SHA256
    ):
        raise ValueError("optimizer applicability repair source or target is not allowlisted")
    if contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V3 and (
        commit != CANONICAL_LF_PACKAGING_SOURCE_COMMIT
        or target_recipe_version != CANONICAL_LF_PACKAGING_TARGET_RECIPE_VERSION
        or _require_sha256(
            value.get("source_runner_expected_sha256"),
            field="source_runner_expected_sha256",
        )
        != CANONICAL_LF_PACKAGING_SOURCE_EXPECTED_RUNNER_SHA256
        or _require_sha256(
            value.get("source_runner_observed_sha256"),
            field="source_runner_observed_sha256",
        )
        != CANONICAL_LF_PACKAGING_SOURCE_OBSERVED_RUNNER_SHA256
        or _require_sha256(
            value.get(TRANSPARENT_BASELINE_RUNNER_FIELD),
            field=TRANSPARENT_BASELINE_RUNNER_FIELD,
        )
        != CANONICAL_LF_PACKAGING_TARGET_RUNNER_SHA256
        or value.get("packaging_contract_version")
        != CANONICAL_LF_PACKAGING_CONTRACT_VERSION
    ):
        raise ValueError("canonical LF packaging repair source or target is not allowlisted")
    if contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V4 and (
        commit != RUNTIME_ALIGNMENT_SOURCE_COMMIT
        or target_recipe_version != RUNTIME_ALIGNMENT_TARGET_RECIPE_VERSION
        or _require_sha256(
            value.get("source_runner_sha256"), field="source_runner_sha256"
        )
        != RUNTIME_ALIGNMENT_SOURCE_RUNNER_SHA256
        or _require_sha256(
            value.get(TRANSPARENT_BASELINE_RUNNER_FIELD),
            field=TRANSPARENT_BASELINE_RUNNER_FIELD,
        )
        != RUNTIME_ALIGNMENT_TARGET_RUNNER_SHA256
        or _require_sha256(
            value.get("source_runtime_bundle_sha256"),
            field="source_runtime_bundle_sha256",
        )
        != RUNTIME_ALIGNMENT_SOURCE_BUNDLE_SHA256
        or _require_sha256(
            value.get("target_runtime_bundle_sha256"),
            field="target_runtime_bundle_sha256",
        )
        != RUNTIME_ALIGNMENT_TARGET_BUNDLE_SHA256
        or value.get("runtime_contract_version")
        != RUNTIME_ALIGNMENT_CONTRACT_VERSION
    ):
        raise ValueError("runtime alignment repair source or target is not allowlisted")
    if contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V5 and (
        commit != RUNTIME_INPUT_SCOPE_SOURCE_COMMIT
        or target_recipe_version != RUNTIME_INPUT_SCOPE_TARGET_RECIPE_VERSION
        or _require_sha256(
            value.get("source_batch_sha256"), field="source_batch_sha256"
        )
        != RUNTIME_INPUT_SCOPE_SOURCE_BATCH_SHA256
        or _require_sha256(
            value.get("source_dataset_identity_sha256"),
            field="source_dataset_identity_sha256",
        )
        != RUNTIME_INPUT_SCOPE_SOURCE_DATASET_IDENTITY_SHA256
        or _require_sha256(
            value.get("source_dataset_lineage_id"),
            field="source_dataset_lineage_id",
        )
        != RUNTIME_INPUT_SCOPE_SOURCE_DATASET_LINEAGE_ID
        or _require_sha256(
            value.get("source_runner_sha256"), field="source_runner_sha256"
        )
        != RUNTIME_INPUT_SCOPE_SOURCE_RUNNER_SHA256
        or _require_sha256(
            value.get(TRANSPARENT_BASELINE_RUNNER_FIELD),
            field=TRANSPARENT_BASELINE_RUNNER_FIELD,
        )
        != RUNTIME_INPUT_SCOPE_TARGET_RUNNER_SHA256
        or _require_sha256(
            value.get("source_runtime_bundle_sha256"),
            field="source_runtime_bundle_sha256",
        )
        != RUNTIME_INPUT_SCOPE_SOURCE_BUNDLE_SHA256
        or _require_sha256(
            value.get("target_runtime_bundle_sha256"),
            field="target_runtime_bundle_sha256",
        )
        != RUNTIME_INPUT_SCOPE_TARGET_BUNDLE_SHA256
        or value.get("runtime_contract_version")
        != RUNTIME_INPUT_SCOPE_CONTRACT_VERSION
    ):
        raise ValueError("runtime input-scope repair source or target is not allowlisted")
    if contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V5 and (
        not isinstance(value.get("target_change_codes"), list)
        or value.get("target_change_codes")
        != list(RUNTIME_INPUT_SCOPE_TARGET_CHANGE_CODES)
    ):
        raise ValueError("runtime input-scope target changes are not allowlisted")
    if contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V6 and (
        commit != FILL_AWARE_HOLDING_AGE_SOURCE_COMMIT
        or target_recipe_version != FILL_AWARE_HOLDING_AGE_TARGET_RECIPE_VERSION
        or _require_sha256(
            value.get("source_batch_sha256"), field="source_batch_sha256"
        )
        != FILL_AWARE_HOLDING_AGE_SOURCE_BATCH_SHA256
        or _require_sha256(
            value.get("source_dataset_identity_sha256"),
            field="source_dataset_identity_sha256",
        )
        != FILL_AWARE_HOLDING_AGE_SOURCE_DATASET_IDENTITY_SHA256
        or _require_sha256(
            value.get("source_dataset_lineage_id"),
            field="source_dataset_lineage_id",
        )
        != FILL_AWARE_HOLDING_AGE_SOURCE_DATASET_LINEAGE_ID
        or _require_sha256(
            value.get("source_runner_sha256"), field="source_runner_sha256"
        )
        != FILL_AWARE_HOLDING_AGE_SOURCE_RUNNER_SHA256
        or _require_sha256(
            value.get(TRANSPARENT_BASELINE_RUNNER_FIELD),
            field=TRANSPARENT_BASELINE_RUNNER_FIELD,
        )
        != FILL_AWARE_HOLDING_AGE_TARGET_RUNNER_SHA256
        or _require_sha256(
            value.get("source_runtime_bundle_sha256"),
            field="source_runtime_bundle_sha256",
        )
        != FILL_AWARE_HOLDING_AGE_SOURCE_BUNDLE_SHA256
        or _require_sha256(
            value.get("target_runtime_bundle_sha256"),
            field="target_runtime_bundle_sha256",
        )
        != FILL_AWARE_HOLDING_AGE_TARGET_BUNDLE_SHA256
        or value.get("runtime_contract_version")
        != FILL_AWARE_HOLDING_AGE_RUNTIME_CONTRACT_VERSION
        or _require_sha256(
            value.get("source_artifact_inventories_sha256"),
            field="source_artifact_inventories_sha256",
        )
        != FILL_AWARE_HOLDING_AGE_SOURCE_ARTIFACT_INVENTORIES_SHA256
        or _require_sha256(
            value.get("source_unopened_history_selection_sha256"),
            field="source_unopened_history_selection_sha256",
        )
        != FILL_AWARE_HOLDING_AGE_SOURCE_SELECTION_SHA256
        or _require_sha256(
            value.get("source_unavailable_horizons_sha256"),
            field="source_unavailable_horizons_sha256",
        )
        != FILL_AWARE_HOLDING_AGE_SOURCE_UNAVAILABLE_HORIZONS_SHA256
        or value.get("source_unavailable_evidence_sha256s")
        != sorted(FILL_AWARE_HOLDING_AGE_UNAVAILABLE_EVIDENCE_SHA256S)
        or value.get("target_change_codes")
        != list(FILL_AWARE_HOLDING_AGE_TARGET_CHANGE_CODES)
    ):
        raise ValueError("fill-aware holding-age repair source or target is not allowlisted")
    if contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V7 and (
        commit != DISCRETE_MAX_POSITION_SOURCE_COMMIT
        or target_recipe_version != DISCRETE_MAX_POSITION_TARGET_RECIPE_VERSION
        or _require_sha256(
            value.get("source_batch_sha256"), field="source_batch_sha256"
        )
        != DISCRETE_MAX_POSITION_SOURCE_BATCH_SHA256
        or _require_sha256(
            value.get("source_dataset_identity_sha256"),
            field="source_dataset_identity_sha256",
        )
        != DISCRETE_MAX_POSITION_SOURCE_DATASET_IDENTITY_SHA256
        or _require_sha256(
            value.get("source_dataset_lineage_id"),
            field="source_dataset_lineage_id",
        )
        != DISCRETE_MAX_POSITION_SOURCE_DATASET_LINEAGE_ID
        or _require_sha256(
            value.get("source_runner_sha256"), field="source_runner_sha256"
        )
        != DISCRETE_MAX_POSITION_SOURCE_RUNNER_SHA256
        or _require_sha256(
            value.get(TRANSPARENT_BASELINE_RUNNER_FIELD),
            field=TRANSPARENT_BASELINE_RUNNER_FIELD,
        )
        != DISCRETE_MAX_POSITION_TARGET_RUNNER_SHA256
        or _require_sha256(
            value.get("source_runtime_bundle_sha256"),
            field="source_runtime_bundle_sha256",
        )
        != DISCRETE_MAX_POSITION_SOURCE_BUNDLE_SHA256
        or _require_sha256(
            value.get("target_runtime_bundle_sha256"),
            field="target_runtime_bundle_sha256",
        )
        != DISCRETE_MAX_POSITION_TARGET_BUNDLE_SHA256
        or value.get("runtime_contract_version")
        != DISCRETE_MAX_POSITION_RUNTIME_CONTRACT_VERSION
        or _require_sha256(
            value.get("source_artifact_inventories_sha256"),
            field="source_artifact_inventories_sha256",
        )
        != DISCRETE_MAX_POSITION_SOURCE_ARTIFACT_INVENTORIES_SHA256
        or _require_sha256(
            value.get("source_unopened_history_selection_sha256"),
            field="source_unopened_history_selection_sha256",
        )
        != DISCRETE_MAX_POSITION_SOURCE_SELECTION_SHA256
        or _require_sha256(
            value.get("source_unavailable_horizons_sha256"),
            field="source_unavailable_horizons_sha256",
        )
        != DISCRETE_MAX_POSITION_SOURCE_UNAVAILABLE_HORIZONS_SHA256
        or value.get("source_unavailable_evidence_sha256s")
        != sorted(DISCRETE_MAX_POSITION_UNAVAILABLE_EVIDENCE_SHA256S)
        or value.get("target_change_codes")
        != list(DISCRETE_MAX_POSITION_TARGET_CHANGE_CODES)
    ):
        raise ValueError("discrete max-position repair source or target is not allowlisted")
    if contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V8 and (
        commit != TOPK_INDUSTRY_CAPACITY_SOURCE_COMMIT
        or target_recipe_version != TOPK_INDUSTRY_CAPACITY_TARGET_RECIPE_VERSION
        or _require_sha256(
            value.get("source_batch_sha256"), field="source_batch_sha256"
        )
        != TOPK_INDUSTRY_CAPACITY_SOURCE_BATCH_SHA256
        or _require_sha256(
            value.get("source_dataset_identity_sha256"),
            field="source_dataset_identity_sha256",
        )
        != TOPK_INDUSTRY_CAPACITY_SOURCE_DATASET_IDENTITY_SHA256
        or _require_sha256(
            value.get("source_dataset_lineage_id"),
            field="source_dataset_lineage_id",
        )
        != TOPK_INDUSTRY_CAPACITY_SOURCE_DATASET_LINEAGE_ID
        or _require_sha256(
            value.get("source_runner_sha256"), field="source_runner_sha256"
        )
        != TOPK_INDUSTRY_CAPACITY_SOURCE_RUNNER_SHA256
        or _require_sha256(
            value.get(TRANSPARENT_BASELINE_RUNNER_FIELD),
            field=TRANSPARENT_BASELINE_RUNNER_FIELD,
        )
        != TOPK_INDUSTRY_CAPACITY_TARGET_RUNNER_SHA256
        or _require_sha256(
            value.get("source_runtime_bundle_sha256"),
            field="source_runtime_bundle_sha256",
        )
        != TOPK_INDUSTRY_CAPACITY_SOURCE_BUNDLE_SHA256
        or _require_sha256(
            value.get("target_runtime_bundle_sha256"),
            field="target_runtime_bundle_sha256",
        )
        != TOPK_INDUSTRY_CAPACITY_TARGET_BUNDLE_SHA256
        or value.get("runtime_contract_version")
        != TOPK_INDUSTRY_CAPACITY_RUNTIME_CONTRACT_VERSION
        or _require_sha256(
            value.get("source_artifact_inventories_sha256"),
            field="source_artifact_inventories_sha256",
        )
        != TOPK_INDUSTRY_CAPACITY_SOURCE_ARTIFACT_INVENTORIES_SHA256
        or _require_sha256(
            value.get("source_unopened_history_selection_sha256"),
            field="source_unopened_history_selection_sha256",
        )
        != TOPK_INDUSTRY_CAPACITY_SOURCE_SELECTION_SHA256
        or _require_sha256(
            value.get("source_unavailable_horizons_sha256"),
            field="source_unavailable_horizons_sha256",
        )
        != TOPK_INDUSTRY_CAPACITY_SOURCE_UNAVAILABLE_HORIZONS_SHA256
        or value.get("source_unavailable_evidence_sha256s")
        != sorted(TOPK_INDUSTRY_CAPACITY_UNAVAILABLE_EVIDENCE_SHA256S)
        or value.get("target_change_codes")
        != list(TOPK_INDUSTRY_CAPACITY_TARGET_CHANGE_CODES)
    ):
        raise ValueError("topk industry-capacity repair source or target is not allowlisted")
    if (
        value.get("target_eligibility_contract") != ELIGIBILITY_CONTRACT_VERSION
        or value.get("target_stock_scope_contract")
        != GOVERNED_DAILY_STOCK_SCOPE_VERSION
    ):
        raise ValueError("transparent baseline repair target data contract changed")
    reason_codes = value.get("reason_codes")
    if (
        not isinstance(reason_codes, list)
        or set(reason_codes) != expected_reasons
        or len(reason_codes) != len(expected_reasons)
        or value.get("performance_information_used") is not False
    ):
        raise ValueError("transparent baseline repair reason or performance boundary is invalid")
    raw_members = value.get("members")
    expected_member_count = (
        1
        if contract_version
        in {
            PRE_RESULT_REPAIR_CONTRACT_VERSION_V6,
            PRE_RESULT_REPAIR_CONTRACT_VERSION_V7,
            PRE_RESULT_REPAIR_CONTRACT_VERSION_V8,
        }
        else 3
    )
    if not isinstance(raw_members, list) or len(raw_members) != expected_member_count:
        raise ValueError(
            "transparent baseline repair must seal exactly "
            f"{expected_member_count} attempt{'s' if expected_member_count != 1 else ''}"
        )
    members: list[dict[str, Any]] = []
    identifiers: dict[str, set[str]] = {
        "backtest_id": set(),
        "job_id": set(),
        "strategy_version_id": set(),
    }
    observed_failure_markers: set[str] = set()
    member_keys = {
        "backtest_id",
        "strategy_version_id",
        "job_id",
        "dataset",
        "periods",
        "status",
        "job_status",
        "error",
        "metrics_absent",
        "result_absent",
        "files",
    }
    for raw in raw_members:
        if not isinstance(raw, Mapping) or set(raw) != member_keys:
            raise ValueError("transparent baseline repair member fields are invalid")
        member = dict(raw)
        for field in identifiers:
            identifier = _require_identifier(member.get(field), field=field)
            if identifier in identifiers[field]:
                raise ValueError(f"transparent baseline repair repeats {field}")
            identifiers[field].add(identifier)
            member[field] = identifier
        dataset = str(member.get("dataset") or "").strip()
        if not dataset or Path(dataset).name != dataset:
            raise ValueError("transparent baseline repair dataset is invalid")
        member["dataset"] = dataset
        periods = member.get("periods")
        if not isinstance(periods, Mapping) or set(periods) != {
            "historical_start",
            "historical_end",
            "start",
            "end",
        }:
            raise ValueError("transparent baseline repair periods are invalid")
        try:
            dates = {key: date.fromisoformat(str(periods[key])) for key in periods}
        except (TypeError, ValueError) as exc:
            raise ValueError("transparent baseline repair periods are invalid") from exc
        if not (
            dates["historical_start"]
            <= dates["historical_end"]
            < dates["start"]
            <= dates["end"]
        ):
            raise ValueError("transparent baseline repair periods are not ordered")
        member["periods"] = {key: dates[key].isoformat() for key in periods}
        if (
            member.get("metrics_absent") is not True
            or member.get("result_absent") is not True
            or member.get("status") not in {"running", "failed"}
            or member.get("job_status") != member.get("status")
        ):
            raise ValueError("transparent baseline repair member had a result or invalid state")
        if contract_version in {
            PRE_RESULT_REPAIR_CONTRACT_VERSION_V2,
            PRE_RESULT_REPAIR_CONTRACT_VERSION_V3,
            PRE_RESULT_REPAIR_CONTRACT_VERSION_V4,
            PRE_RESULT_REPAIR_CONTRACT_VERSION_V5,
            PRE_RESULT_REPAIR_CONTRACT_VERSION_V6,
            PRE_RESULT_REPAIR_CONTRACT_VERSION_V7,
            PRE_RESULT_REPAIR_CONTRACT_VERSION_V8,
        } and (
            member.get("status") != "failed"
        ):
            if contract_version in {
                PRE_RESULT_REPAIR_CONTRACT_VERSION_V6,
                PRE_RESULT_REPAIR_CONTRACT_VERSION_V7,
                PRE_RESULT_REPAIR_CONTRACT_VERSION_V8,
            }:
                raise ValueError("allowlisted repair requires one failed attempt")
            raise ValueError("allowlisted repair requires three failed attempts")
        error = member.get("error")
        if error is not None:
            error_text = str(error)
            if (
                contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V2
                and error_text != OPTIMIZER_APPLICABILITY_ERROR
            ):
                raise ValueError("optimizer applicability repair error is not exact")
            if (
                contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V3
                and error_text != CANONICAL_LF_PACKAGING_ERROR
            ):
                raise ValueError("canonical LF packaging repair error is not exact")
            if contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V4:
                source = RUNTIME_ALIGNMENT_SOURCE_BINDINGS.get(
                    str(member.get("backtest_id") or "")
                )
                if source is None or error_text != source["error"]:
                    raise ValueError("runtime alignment repair error is not exact")
            if contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V5:
                source = RUNTIME_INPUT_SCOPE_SOURCE_BINDINGS.get(
                    str(member.get("backtest_id") or "")
                )
                if source is None or error_text != source["error"]:
                    raise ValueError("runtime input-scope repair error is not exact")
            if contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V6:
                source = FILL_AWARE_HOLDING_AGE_SOURCE_BINDINGS.get(
                    str(member.get("backtest_id") or "")
                )
                if source is None or error_text != source["error"]:
                    raise ValueError("fill-aware holding-age repair error is not exact")
            if contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V7:
                source = DISCRETE_MAX_POSITION_SOURCE_BINDINGS.get(
                    str(member.get("backtest_id") or "")
                )
                if source is None or error_text != source["error"]:
                    raise ValueError("discrete max-position repair error is not exact")
            if contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V8:
                source = TOPK_INDUSTRY_CAPACITY_SOURCE_BINDINGS.get(
                    str(member.get("backtest_id") or "")
                )
                if source is None or error_text != source["error"]:
                    raise ValueError("topk industry-capacity repair error is not exact")
            matches = {
                code
                for code, marker in failure_markers.items()
                if marker in error_text
            }
            if len(matches) != 1 or member.get("status") != "failed":
                raise ValueError("transparent baseline repair failure is not allowlisted")
            observed_failure_markers.update(matches)
            member["error"] = error_text
        elif member.get("status") != "running":
            raise ValueError("transparent baseline repair failed member has no error")
        files = member.get("files")
        if not isinstance(files, list):
            raise ValueError("transparent baseline repair artifact inventory is missing")
        if contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V3 and files:
            raise ValueError("canonical LF packaging repair artifacts must be empty")
        if contract_version != PRE_RESULT_REPAIR_CONTRACT_VERSION_V3 and not files:
            raise ValueError("transparent baseline repair artifact inventory is missing")
        normalized_files: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        for item in files:
            if not isinstance(item, Mapping) or set(item) != {"path", "bytes", "sha256"}:
                raise ValueError("transparent baseline repair artifact fields are invalid")
            path = str(item.get("path") or "").replace("\\", "/").strip("/")
            if (
                not path
                or path in seen_paths
                or (path != "manifest.json" and not path.startswith("baseline/"))
                or Path(path).is_absolute()
                or ".." in Path(path).parts
            ):
                raise ValueError("transparent baseline repair contains a result artifact")
            size = item.get("bytes")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ValueError("transparent baseline repair artifact size is invalid")
            seen_paths.add(path)
            normalized_files.append(
                {
                    "path": path,
                    "bytes": size,
                    "sha256": _require_sha256(
                        item.get("sha256"), field="artifact sha256"
                    ),
                }
            )
        if (
            contract_version != PRE_RESULT_REPAIR_CONTRACT_VERSION_V3
            and "manifest.json" not in seen_paths
        ):
            raise ValueError("transparent baseline repair execution manifest is missing")
        member["files"] = normalized_files
        members.append(member)
    if contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V2 and (
        identifiers["backtest_id"] != OPTIMIZER_APPLICABILITY_SOURCE_BACKTEST_IDS
    ):
        raise ValueError("optimizer applicability repair backtests are not allowlisted")
    if contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V3 and (
        identifiers["backtest_id"] != CANONICAL_LF_PACKAGING_SOURCE_BACKTEST_IDS
    ):
        raise ValueError("canonical LF packaging repair backtests are not allowlisted")
    if contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V3:
        for member in members:
            binding = CANONICAL_LF_PACKAGING_SOURCE_BINDINGS.get(
                member["backtest_id"]
            )
            if binding is None or any(
                member[field] != binding[field]
                for field in ("job_id", "strategy_version_id")
            ):
                raise ValueError("canonical LF packaging source binding changed")
    if contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V4:
        if identifiers["backtest_id"] != RUNTIME_ALIGNMENT_SOURCE_BACKTEST_IDS:
            raise ValueError("runtime alignment repair backtests are not allowlisted")
        inventory_digests: list[dict[str, str]] = []
        for member in members:
            source = RUNTIME_ALIGNMENT_SOURCE_BINDINGS.get(member["backtest_id"])
            if source is None or any(
                member[field] != source[field]
                for field in ("job_id", "strategy_version_id")
            ):
                raise ValueError("runtime alignment source binding changed")
            if (
                member["dataset"] != RUNTIME_ALIGNMENT_SOURCE_DATASET
                or member["periods"] != source["periods"]
                or member["error"] != source["error"]
            ):
                raise ValueError("runtime alignment source evidence changed")
            inventory_sha256 = canonical_sha256(member["files"])
            if inventory_sha256 != source["artifact_inventory_sha256"]:
                raise ValueError("runtime alignment artifact inventory changed")
            inventory_digests.append(
                {
                    "backtest_id": member["backtest_id"],
                    "artifact_inventory_sha256": inventory_sha256,
                }
            )
        if _require_sha256(
            value.get("source_artifact_inventories_sha256"),
            field="source_artifact_inventories_sha256",
        ) != canonical_sha256(
            sorted(inventory_digests, key=lambda item: item["backtest_id"])
        ):
            raise ValueError("runtime alignment aggregate artifact inventory changed")
    if contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V5:
        if identifiers["backtest_id"] != RUNTIME_INPUT_SCOPE_SOURCE_BACKTEST_IDS:
            raise ValueError("runtime input-scope repair backtests are not allowlisted")
        inventory_digests: list[dict[str, str]] = []
        for member in members:
            source = RUNTIME_INPUT_SCOPE_SOURCE_BINDINGS.get(member["backtest_id"])
            if source is None or any(
                member[field] != source[field]
                for field in ("job_id", "strategy_version_id")
            ):
                raise ValueError("runtime input-scope source binding changed")
            if (
                member["dataset"] != RUNTIME_INPUT_SCOPE_SOURCE_DATASET
                or member["periods"] != source["periods"]
                or member["error"] != source["error"]
            ):
                raise ValueError("runtime input-scope source evidence changed")
            inventory_sha256 = canonical_sha256(member["files"])
            if inventory_sha256 != source["artifact_inventory_sha256"]:
                raise ValueError("runtime input-scope artifact inventory changed")
            inventory_digests.append(
                {
                    "backtest_id": member["backtest_id"],
                    "artifact_inventory_sha256": inventory_sha256,
                }
            )
        aggregate_inventory = canonical_sha256(
            sorted(inventory_digests, key=lambda item: item["backtest_id"])
        )
        if (
            aggregate_inventory
            != RUNTIME_INPUT_SCOPE_SOURCE_ARTIFACT_INVENTORIES_SHA256
            or _require_sha256(
                value.get("source_artifact_inventories_sha256"),
                field="source_artifact_inventories_sha256",
            )
            != aggregate_inventory
        ):
            raise ValueError("runtime input-scope aggregate artifact inventory changed")
    if contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V6:
        if identifiers["backtest_id"] != FILL_AWARE_HOLDING_AGE_SOURCE_BACKTEST_IDS:
            raise ValueError("fill-aware holding-age repair backtest is not allowlisted")
        member = members[0]
        source = FILL_AWARE_HOLDING_AGE_SOURCE_BINDINGS.get(member["backtest_id"])
        if source is None or any(
            member[field] != source[field]
            for field in ("job_id", "strategy_version_id")
        ):
            raise ValueError("fill-aware holding-age source binding changed")
        if (
            member["dataset"] != FILL_AWARE_HOLDING_AGE_SOURCE_DATASET
            or member["periods"] != source["periods"]
            or member["error"] != source["error"]
        ):
            raise ValueError("fill-aware holding-age source evidence changed")
        inventory_sha256 = canonical_sha256(member["files"])
        inventory_rows = [
            {
                "backtest_id": member["backtest_id"],
                "artifact_inventory_sha256": inventory_sha256,
            }
        ]
        if (
            inventory_sha256 != source["artifact_inventory_sha256"]
            or canonical_sha256(inventory_rows)
            != FILL_AWARE_HOLDING_AGE_SOURCE_ARTIFACT_INVENTORIES_SHA256
        ):
            raise ValueError("fill-aware holding-age artifact inventory changed")
    if contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V7:
        if identifiers["backtest_id"] != DISCRETE_MAX_POSITION_SOURCE_BACKTEST_IDS:
            raise ValueError("discrete max-position repair backtest is not allowlisted")
        member = members[0]
        source = DISCRETE_MAX_POSITION_SOURCE_BINDINGS.get(member["backtest_id"])
        if source is None or any(
            member[field] != source[field]
            for field in ("job_id", "strategy_version_id")
        ):
            raise ValueError("discrete max-position source binding changed")
        if (
            member["dataset"] != DISCRETE_MAX_POSITION_SOURCE_DATASET
            or member["periods"] != source["periods"]
            or member["error"] != source["error"]
        ):
            raise ValueError("discrete max-position source evidence changed")
        inventory_sha256 = canonical_sha256(member["files"])
        inventory_rows = [
            {
                "backtest_id": member["backtest_id"],
                "artifact_inventory_sha256": inventory_sha256,
            }
        ]
        if (
            inventory_sha256 != source["artifact_inventory_sha256"]
            or canonical_sha256(inventory_rows)
            != DISCRETE_MAX_POSITION_SOURCE_ARTIFACT_INVENTORIES_SHA256
        ):
            raise ValueError("discrete max-position artifact inventory changed")
    if contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V8:
        if identifiers["backtest_id"] != TOPK_INDUSTRY_CAPACITY_SOURCE_BACKTEST_IDS:
            raise ValueError("topk industry-capacity repair backtest is not allowlisted")
        member = members[0]
        source = TOPK_INDUSTRY_CAPACITY_SOURCE_BINDINGS.get(member["backtest_id"])
        if source is None or any(
            member[field] != source[field]
            for field in ("job_id", "strategy_version_id")
        ):
            raise ValueError("topk industry-capacity source binding changed")
        if (
            member["dataset"] != TOPK_INDUSTRY_CAPACITY_SOURCE_DATASET
            or member["periods"] != source["periods"]
            or member["error"] != source["error"]
        ):
            raise ValueError("topk industry-capacity source evidence changed")
        inventory_sha256 = canonical_sha256(member["files"])
        inventory_rows = [
            {
                "backtest_id": member["backtest_id"],
                "artifact_inventory_sha256": inventory_sha256,
            }
        ]
        if (
            inventory_sha256 != source["artifact_inventory_sha256"]
            or canonical_sha256(inventory_rows)
            != TOPK_INDUSTRY_CAPACITY_SOURCE_ARTIFACT_INVENTORIES_SHA256
        ):
            raise ValueError("topk industry-capacity artifact inventory changed")
    if observed_failure_markers != expected_reasons:
        raise ValueError("transparent baseline repair does not cover its allowlisted defect")
    return {
        **payload,
        "source_release_commit": commit,
        "target_recipe_version": target_recipe_version,
        **({"repair_generation": repair_generation} if repair_generation else {}),
        "reason_codes": list(reason_codes),
        "members": members,
        "receipt_sha256": receipt_sha256,
    }


def _normalize_member(raw: Mapping[str, Any]) -> dict[str, str]:
    recipe_id = str(raw.get("recipe_id") or "").strip()
    if recipe_id not in _RECIPE_HORIZONS:
        raise ValueError("transparent baseline lockbox contains an unknown recipe")
    horizon = str(raw.get("horizon_profile") or "").strip()
    if horizon != _RECIPE_HORIZONS[recipe_id]:
        raise ValueError("transparent baseline lockbox recipe horizon changed")
    recipe_version = str(raw.get("recipe_version") or "").strip()
    if not recipe_version:
        raise ValueError("transparent baseline lockbox recipe version is required")
    expected_runner = target_runner_for_recipe(recipe_id, recipe_version)
    expected_runtime_bundle = target_runtime_bundle_for_recipe(
        recipe_id, recipe_version
    )
    expected_keys = (
        _RUNTIME_BOUND_MEMBER_KEYS
        if expected_runtime_bundle
        else _RUNNER_BOUND_MEMBER_KEYS
        if expected_runner
        else _MEMBER_KEYS
    )
    if set(raw) != expected_keys:
        raise ValueError("transparent baseline lockbox member fields are invalid")
    try:
        historical_start = date.fromisoformat(str(raw["historical_start"]))
        historical_end = date.fromisoformat(str(raw["historical_end"]))
        test_start = date.fromisoformat(str(raw["test_start"]))
        test_end = date.fromisoformat(str(raw["test_end"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("transparent baseline lockbox periods are invalid") from exc
    if not historical_start <= historical_end < test_start <= test_end:
        raise ValueError("transparent baseline lockbox periods are not ordered")
    member = {
        "recipe_id": recipe_id,
        "recipe_version": recipe_version,
        "recipe_sha256": _require_sha256(
            raw.get("recipe_sha256"), field="recipe_sha256"
        ),
        "horizon_profile": horizon,
        "base_config_sha256": _require_sha256(
            raw.get("base_config_sha256"), field="base_config_sha256"
        ),
        "baseline_definition_sha256": _require_sha256(
            raw.get("baseline_definition_sha256"),
            field="baseline_definition_sha256",
        ),
        "research_window_contract_sha256": _require_sha256(
            raw.get("research_window_contract_sha256"),
            field="research_window_contract_sha256",
        ),
        "historical_start": historical_start.isoformat(),
        "historical_end": historical_end.isoformat(),
        "test_start": test_start.isoformat(),
        "test_end": test_end.isoformat(),
    }
    if expected_runner is not None:
        member[TRANSPARENT_BASELINE_RUNNER_FIELD] = _require_sha256(
            raw.get(TRANSPARENT_BASELINE_RUNNER_FIELD),
            field=TRANSPARENT_BASELINE_RUNNER_FIELD,
        )
        if member[TRANSPARENT_BASELINE_RUNNER_FIELD] != expected_runner:
            raise ValueError("transparent lockbox runner identity changed")
    if expected_runtime_bundle is not None:
        member[TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD] = _require_sha256(
            raw.get(TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD),
            field=TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD,
        )
        if (
            member[TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD]
            != expected_runtime_bundle
        ):
            raise ValueError("transparent lockbox runtime bundle identity changed")
    return member


def _normalize_unavailable_horizon(value: Mapping[str, Any]) -> dict[str, Any]:
    keys = {
        "recipe_id",
        "horizon_profile",
        "status",
        "reason",
        "evidence",
        "evidence_sha256",
    }
    if set(value) != keys:
        raise ValueError("unavailable baseline horizon contract is invalid")
    recipe_id = str(value.get("recipe_id") or "")
    horizon_profile = str(value.get("horizon_profile") or "")
    evidence = value.get("evidence")
    if (
        recipe_id not in _RECIPE_HORIZONS
        or horizon_profile != _RECIPE_HORIZONS[recipe_id]
        or value.get("status") != "unavailable"
        or not str(value.get("reason") or "").strip()
        or not isinstance(evidence, Mapping)
        or evidence.get("capital_evaluation_eligible") is not False
        or not str(
            evidence.get("capital_evaluation_unavailable_reason") or ""
        ).strip()
        or canonical_sha256(dict(evidence))
        != _require_sha256(value.get("evidence_sha256"), field="evidence_sha256")
    ):
        raise ValueError("unavailable baseline horizon evidence is invalid")
    return {
        "recipe_id": recipe_id,
        "horizon_profile": horizon_profile,
        "status": "unavailable",
        "reason": str(value["reason"]).strip(),
        "evidence": dict(evidence),
        "evidence_sha256": canonical_sha256(dict(evidence)),
    }


def build_joint_lockbox(
    *,
    dataset: str,
    dataset_identity_sha256: str,
    dataset_lineage_id: str,
    members: Sequence[Mapping[str, Any]],
    unopened_history_selection: Mapping[str, Any] | None = None,
    unavailable_horizons: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build one atomic preregistration for every statistically available lane."""

    normalized_members = sorted(
        (_normalize_member(item) for item in members),
        key=lambda item: item["recipe_id"],
    )
    normalized_unavailable = sorted(
        (
            _normalize_unavailable_horizon(item)
            for item in (unavailable_horizons or [])
        ),
        key=lambda item: item["recipe_id"],
    )
    member_recipe_ids = [str(item["recipe_id"]) for item in normalized_members]
    unavailable_recipe_ids = [
        str(item["recipe_id"]) for item in normalized_unavailable
    ]
    expected_recipe_ids = sorted(TRANSPARENT_RESEARCH_BASELINE_IDS)
    if (
        not normalized_members
        or len(member_recipe_ids) != len(set(member_recipe_ids))
        or len({item["horizon_profile"] for item in normalized_members})
        != len(normalized_members)
        or sorted(member_recipe_ids + unavailable_recipe_ids) != expected_recipe_ids
    ):
        raise ValueError(
            "baseline lockbox must account exactly once for all three public horizons"
        )
    dataset_name = str(dataset or "").strip()
    if not dataset_name:
        raise ValueError("joint lockbox dataset is required")
    selection = (
        validate_unopened_history_selection(unopened_history_selection)
        if unopened_history_selection is not None
        else None
    )
    if selection is not None:
        member_recipe_versions = {
            str(item["recipe_version"]) for item in normalized_members
        }
        if member_recipe_versions != {selection["current_recipe_version"]}:
            raise ValueError(
                "joint lockbox history selection differs from its recipe version"
            )
    if normalized_unavailable and selection is None:
        raise ValueError("partial baseline lockbox requires unopened-history evidence")
    contract_version = (
        LOCKBOX_CONTRACT_VERSION_V3
        if normalized_unavailable
        else LOCKBOX_CONTRACT_VERSION_V2
        if selection is not None
        else LOCKBOX_CONTRACT_VERSION
    )
    contract = {
        "contract_version": contract_version,
        "dataset": dataset_name,
        "dataset_identity_sha256": _require_sha256(
            dataset_identity_sha256,
            field="dataset_identity_sha256",
        ),
        "dataset_lineage_id": _require_sha256(
            dataset_lineage_id,
            field="dataset_lineage_id",
        ),
        "members": normalized_members,
        **(
            {"unopened_history_selection": selection}
            if selection is not None
            else {}
        ),
        **(
            {"unavailable_horizons": normalized_unavailable}
            if normalized_unavailable
            else {}
        ),
    }
    return {**contract, "batch_sha256": canonical_sha256(contract)}


def validate_joint_lockbox(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("transparent baseline joint lockbox is required")
    contract_version = value.get("contract_version")
    allowed = {
        "contract_version",
        "dataset",
        "dataset_identity_sha256",
        "dataset_lineage_id",
        "members",
        "batch_sha256",
    }
    if contract_version in {
        LOCKBOX_CONTRACT_VERSION_V2,
        LOCKBOX_CONTRACT_VERSION_V3,
    }:
        allowed.add("unopened_history_selection")
        if contract_version == LOCKBOX_CONTRACT_VERSION_V3:
            allowed.add("unavailable_horizons")
    elif contract_version != LOCKBOX_CONTRACT_VERSION:
        raise ValueError("transparent baseline joint lockbox contract is invalid")
    if set(value) != allowed:
        raise ValueError("transparent baseline joint lockbox contract is invalid")
    members = value.get("members")
    if not isinstance(members, Sequence) or isinstance(members, (str, bytes)):
        raise ValueError("transparent baseline joint lockbox members are invalid")
    normalized = build_joint_lockbox(
        dataset=str(value.get("dataset") or ""),
        dataset_identity_sha256=str(value.get("dataset_identity_sha256") or ""),
        dataset_lineage_id=str(value.get("dataset_lineage_id") or ""),
        members=[item for item in members if isinstance(item, Mapping)],
        unopened_history_selection=(
            value.get("unopened_history_selection")
            if contract_version
            in {LOCKBOX_CONTRACT_VERSION_V2, LOCKBOX_CONTRACT_VERSION_V3}
            else None
        ),
        unavailable_horizons=(
            value.get("unavailable_horizons")
            if contract_version == LOCKBOX_CONTRACT_VERSION_V3
            else None
        ),
    )
    if len(members) != len(normalized["members"]) or dict(value) != normalized:
        raise ValueError("transparent baseline joint lockbox digest or members changed")
    return normalized


def build_lockbox_member(
    *,
    config: Mapping[str, Any],
    formal_periods: Mapping[str, Any],
) -> dict[str, str]:
    """Describe one normalized StrategySpec before the batch binding is added."""

    bootstrap = config.get(BOOTSTRAP_CONFIG_KEY)
    if not isinstance(bootstrap, Mapping):
        raise ValueError("transparent baseline config has no frozen bootstrap contract")
    try:
        periods = {
            key: date.fromisoformat(str(formal_periods[key])).isoformat()
            for key in ("historical_start", "historical_end", "start", "end")
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("transparent baseline formal periods are invalid") from exc
    if dict(bootstrap.get("formal_periods") or {}) != periods:
        raise ValueError("transparent baseline formal periods differ from its bootstrap")
    raw_member = {
            "recipe_id": config.get("recipe_id"),
            "recipe_version": config.get("recipe_version"),
            "recipe_sha256": bootstrap.get("recipe_sha256"),
            "horizon_profile": config.get("horizon_profile"),
            "base_config_sha256": canonical_sha256(dict(config)),
            "baseline_definition_sha256": config.get(
                "baseline_definition_sha256"
            ),
            "research_window_contract_sha256": bootstrap.get(
                "research_window_contract_sha256"
            ),
            "historical_start": periods["historical_start"],
            "historical_end": periods["historical_end"],
            "test_start": periods["start"],
            "test_end": periods["end"],
    }
    runner_sha256 = bootstrap.get(TRANSPARENT_BASELINE_RUNNER_FIELD)
    if runner_sha256 is not None:
        raw_member[TRANSPARENT_BASELINE_RUNNER_FIELD] = runner_sha256
    runtime_bundle_sha256 = bootstrap.get(TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD)
    if runtime_bundle_sha256 is not None:
        raw_member[TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD] = runtime_bundle_sha256
    return _normalize_member(raw_member)


def lockbox_member_link(config: Mapping[str, Any]) -> dict[str, Any] | None:
    raw = config.get(LOCKBOX_CONFIG_KEY)
    if raw is None:
        return None
    lockbox = validate_joint_lockbox(raw)
    recipe_id = str(config.get("recipe_id") or "")
    matches = [
        item for item in lockbox["members"] if item["recipe_id"] == recipe_id
    ]
    if len(matches) != 1:
        raise ValueError("strategy recipe is not a unique joint-lockbox member")
    member = matches[0]
    base_config = dict(config)
    base_config.pop(LOCKBOX_CONFIG_KEY, None)
    bootstrap = base_config.get(BOOTSTRAP_CONFIG_KEY)
    if not isinstance(bootstrap, Mapping):
        raise ValueError("strategy config has no frozen bootstrap contract")
    if (
        canonical_sha256(base_config) != member["base_config_sha256"]
        or str(config.get("recipe_version") or "") != member["recipe_version"]
        or str(bootstrap.get("recipe_sha256") or "") != member["recipe_sha256"]
        or str(config.get("horizon_profile") or "") != member["horizon_profile"]
        or str(config.get("baseline_definition_sha256") or "")
        != member["baseline_definition_sha256"]
        or str(bootstrap.get("research_window_contract_sha256") or "")
        != member["research_window_contract_sha256"]
        or dict(bootstrap.get("formal_periods") or {})
        != {
            "historical_start": member["historical_start"],
            "historical_end": member["historical_end"],
            "start": member["test_start"],
            "end": member["test_end"],
        }
        or (
            lockbox.get("unopened_history_selection")
            != bootstrap.get("unopened_history_selection")
        )
        or (
            member.get(TRANSPARENT_BASELINE_RUNNER_FIELD)
            != bootstrap.get(TRANSPARENT_BASELINE_RUNNER_FIELD)
        )
        or (
            member.get(TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD)
            != bootstrap.get(TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD)
        )
    ):
        raise ValueError("strategy config differs from its joint-lockbox member")
    member_hashes = sorted(canonical_sha256(item) for item in lockbox["members"])
    return {
        "contract_version": (
            LOCKBOX_LINK_VERSION_V2
            if lockbox["contract_version"] == LOCKBOX_CONTRACT_VERSION_V3
            else LOCKBOX_LINK_VERSION
        ),
        "batch_sha256": lockbox["batch_sha256"],
        "member_sha256": canonical_sha256(member),
        "member_sha256s": member_hashes,
        "recipe_id": recipe_id,
        "horizon_profile": member["horizon_profile"],
    }


def validate_lockbox_link(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("transparent baseline lockbox link is invalid")
    keys = {
        "contract_version",
        "batch_sha256",
        "member_sha256",
        "member_sha256s",
        "recipe_id",
        "horizon_profile",
    }
    link_version = value.get("contract_version")
    if set(value) != keys or link_version not in {
        LOCKBOX_LINK_VERSION,
        LOCKBOX_LINK_VERSION_V2,
    }:
        raise ValueError("transparent baseline lockbox link contract is invalid")
    recipe_id = str(value.get("recipe_id") or "")
    if (
        recipe_id not in _RECIPE_HORIZONS
        or str(value.get("horizon_profile") or "") != _RECIPE_HORIZONS[recipe_id]
    ):
        raise ValueError("transparent baseline lockbox link recipe is invalid")
    member_hashes = value.get("member_sha256s")
    expected_count = 3 if link_version == LOCKBOX_LINK_VERSION else None
    if (
        not isinstance(member_hashes, list)
        or not member_hashes
        or len(member_hashes) > 3
        or (expected_count is not None and len(member_hashes) != expected_count)
    ):
        raise ValueError("transparent baseline lockbox link member count is invalid")
    normalized_hashes = sorted(
        _require_sha256(item, field="member_sha256") for item in member_hashes
    )
    member_sha256 = _require_sha256(
        value.get("member_sha256"), field="member_sha256"
    )
    if (
        len(set(normalized_hashes)) != len(normalized_hashes)
        or member_sha256 not in normalized_hashes
    ):
        raise ValueError("transparent baseline lockbox member identities are invalid")
    return {
        "contract_version": str(link_version),
        "batch_sha256": _require_sha256(
            value.get("batch_sha256"), field="batch_sha256"
        ),
        "member_sha256": member_sha256,
        "member_sha256s": normalized_hashes,
        "recipe_id": recipe_id,
        "horizon_profile": _RECIPE_HORIZONS[recipe_id],
    }


def baseline_oos_sealed_member_set(version: Mapping[str, Any]) -> dict[str, Any]:
    """Recreate the exact pure-baseline seal used by preregistration/backtest."""

    config = version.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("baseline strategy config is invalid")
    baseline_sha256 = _require_sha256(
        config.get("baseline_definition_sha256"),
        field="baseline_definition_sha256",
    )
    strategy_spec = {
        "strategy_type": "multifactor",
        "benchmark": version.get("benchmark"),
        "universe": version.get("universe"),
        "config_sha256": canonical_sha256(config),
        "baseline_definition_sha256": baseline_sha256,
    }
    result: dict[str, Any] = {
        "candidate_ids": [],
        "baseline_definition_sha256": baseline_sha256,
        "strategy_spec_sha256": canonical_sha256(strategy_spec),
        "model_signal": None,
    }
    link = lockbox_member_link(config)
    if link is not None:
        version_id = str(version.get("id") or "").strip()
        if not version_id:
            raise ValueError("joint-lockbox member requires a strategy version id")
        result.update(
            {
                "strategy_version_id": version_id,
                "transparent_baseline_lockbox": link,
            }
        )
    return result


def _repair_economic_config(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only version/data bindings; all economic rules must stay equal."""

    normalized = json.loads(json.dumps(dict(value), ensure_ascii=False))
    for key in ("recipe_version", BOOTSTRAP_CONFIG_KEY, LOCKBOX_CONFIG_KEY):
        normalized.pop(key, None)
    return normalized


def _repair_bootstrap_semantics(value: Mapping[str, Any]) -> dict[str, Any]:
    """Compare every bootstrap semantic except version-derived byte hashes."""

    bootstrap_raw = value.get(BOOTSTRAP_CONFIG_KEY)
    if not isinstance(bootstrap_raw, Mapping):
        raise ValueError("transparent baseline repair bootstrap is missing")
    bootstrap = json.loads(json.dumps(dict(bootstrap_raw), ensure_ascii=False))
    for key in (
        "recipe_version",
        "recipe_sha256",
        TRANSPARENT_BASELINE_RUNNER_FIELD,
        TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD,
        "research_window_contract_sha256",
    ):
        bootstrap.pop(key, None)
    feature_set = bootstrap.get("feature_set")
    if not isinstance(feature_set, dict):
        raise ValueError("transparent baseline repair feature contract is missing")
    feature_set.pop("recipe_version", None)
    feature_set.pop("definition_sha256", None)
    research_window = bootstrap.get("research_window_contract")
    if not isinstance(research_window, dict):
        raise ValueError("transparent baseline repair research window is missing")
    # This digest is derived from the version-bound feature-set definition;
    # all feature expressions and every other research/data/OOS field remain
    # in the comparison below.
    research_window.pop("feature_set_sha256", None)
    return bootstrap


def _artifact_result_exists(artifact_path: Any) -> bool:
    try:
        root = Path(str(artifact_path)).resolve()
    except (OSError, RuntimeError, ValueError):
        return True
    try:
        return any(path.is_file() for path in root.rglob("result.json"))
    except OSError:
        return True


def _artifact_inventory(artifact_path: Any) -> list[dict[str, Any]]:
    try:
        root = Path(str(artifact_path)).resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("transparent baseline repair artifact root is invalid") from exc
    if not root.is_dir():
        raise ValueError("transparent baseline repair artifact root is missing")
    files: list[dict[str, Any]] = []
    try:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": digest.hexdigest(),
                }
            )
    except OSError as exc:
        raise ValueError("transparent baseline repair artifacts cannot be verified") from exc
    return files


def _row_field(row: Any, field: str) -> Any:
    if isinstance(row, Mapping):
        return row.get(field)
    return getattr(row, field, None)


def validate_pre_result_repair_audit_event(
    audit_event: Any,
    *,
    expected_receipt: Mapping[str, Any] | None = None,
    expected_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate both the immutable receipt and its audit-event envelope."""

    failure = "transparent baseline pre-result repair audit event is invalid"
    try:
        created_at = _row_field(audit_event, "created_at")
        if (
            str(_row_field(audit_event, "action") or "") != PRE_RESULT_REPAIR_ACTION
            or str(_row_field(audit_event, "method") or "") != "INTERNAL"
            or str(_row_field(audit_event, "path") or "")
            != "transparent-baseline/pre-result-repair"
            or int(_row_field(audit_event, "status_code")) != 201
            or not isinstance(created_at, datetime)
            or created_at.tzinfo is None
            or created_at.utcoffset() is None
        ):
            raise ValueError(failure)
        receipt = validate_pre_result_repair_receipt(
            dict(_row_field(audit_event, "details_json") or {})
        )
        if (
            expected_receipt_sha256 is not None
            and receipt["receipt_sha256"] != expected_receipt_sha256
        ):
            raise ValueError(failure)
        if expected_receipt is not None and receipt != dict(expected_receipt):
            raise ValueError(failure)
    except (TypeError, ValueError):
        raise ValueError(failure) from None
    return receipt


def _exact_same_lineage_registry_profile(
    verification: Mapping[str, Any],
    source_backtest_ids: set[str],
) -> bool:
    contract_version = verification.get("receipt_contract_version")
    if contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V2:
        return (
            verification.get("repair_generation")
            == OPTIMIZER_APPLICABILITY_REPAIR_GENERATION
            and verification.get(TRANSPARENT_BASELINE_RUNNER_FIELD)
            == OPTIMIZER_APPLICABILITY_TARGET_RUNNER_SHA256
            and verification.get("target_recipe_version")
            == OPTIMIZER_APPLICABILITY_TARGET_RECIPE_VERSION
            and source_backtest_ids
            == set(OPTIMIZER_APPLICABILITY_SOURCE_BACKTEST_IDS)
        )
    if contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V3:
        return (
            verification.get("repair_generation")
            == CANONICAL_LF_PACKAGING_REPAIR_GENERATION
            and verification.get("source_runner_expected_sha256")
            == CANONICAL_LF_PACKAGING_SOURCE_EXPECTED_RUNNER_SHA256
            and verification.get("source_runner_observed_sha256")
            == CANONICAL_LF_PACKAGING_SOURCE_OBSERVED_RUNNER_SHA256
            and verification.get(TRANSPARENT_BASELINE_RUNNER_FIELD)
            == CANONICAL_LF_PACKAGING_TARGET_RUNNER_SHA256
            and verification.get("packaging_contract_version")
            == CANONICAL_LF_PACKAGING_CONTRACT_VERSION
            and verification.get("source_release_commit")
            == CANONICAL_LF_PACKAGING_SOURCE_COMMIT
            and verification.get("target_recipe_version")
            == CANONICAL_LF_PACKAGING_TARGET_RECIPE_VERSION
            and source_backtest_ids
            == set(CANONICAL_LF_PACKAGING_SOURCE_BACKTEST_IDS)
        )
    if contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V4:
        return (
            verification.get("repair_generation")
            == RUNTIME_ALIGNMENT_REPAIR_GENERATION
            and verification.get("source_runner_sha256")
            == RUNTIME_ALIGNMENT_SOURCE_RUNNER_SHA256
            and verification.get(TRANSPARENT_BASELINE_RUNNER_FIELD)
            == RUNTIME_ALIGNMENT_TARGET_RUNNER_SHA256
            and verification.get("source_runtime_bundle_sha256")
            == RUNTIME_ALIGNMENT_SOURCE_BUNDLE_SHA256
            and verification.get("target_runtime_bundle_sha256")
            == RUNTIME_ALIGNMENT_TARGET_BUNDLE_SHA256
            and verification.get("runtime_contract_version")
            == RUNTIME_ALIGNMENT_CONTRACT_VERSION
            and verification.get("source_release_commit")
            == RUNTIME_ALIGNMENT_SOURCE_COMMIT
            and verification.get("target_recipe_version")
            == RUNTIME_ALIGNMENT_TARGET_RECIPE_VERSION
            and verification.get("source_artifact_inventories_sha256")
            == canonical_sha256(
                sorted(
                    (
                        {
                            "backtest_id": backtest_id,
                            "artifact_inventory_sha256": binding[
                                "artifact_inventory_sha256"
                            ],
                        }
                        for backtest_id, binding in RUNTIME_ALIGNMENT_SOURCE_BINDINGS.items()
                    ),
                    key=lambda item: item["backtest_id"],
                )
            )
            and source_backtest_ids == set(RUNTIME_ALIGNMENT_SOURCE_BACKTEST_IDS)
        )
    if contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V5:
        return (
            verification.get("repair_generation")
            == RUNTIME_INPUT_SCOPE_REPAIR_GENERATION
            and verification.get("source_batch_sha256")
            == RUNTIME_INPUT_SCOPE_SOURCE_BATCH_SHA256
            and verification.get("source_dataset_identity_sha256")
            == RUNTIME_INPUT_SCOPE_SOURCE_DATASET_IDENTITY_SHA256
            and verification.get("source_dataset_lineage_id")
            == RUNTIME_INPUT_SCOPE_SOURCE_DATASET_LINEAGE_ID
            and verification.get("source_runner_sha256")
            == RUNTIME_INPUT_SCOPE_SOURCE_RUNNER_SHA256
            and verification.get(TRANSPARENT_BASELINE_RUNNER_FIELD)
            == RUNTIME_INPUT_SCOPE_TARGET_RUNNER_SHA256
            and verification.get("source_runtime_bundle_sha256")
            == RUNTIME_INPUT_SCOPE_SOURCE_BUNDLE_SHA256
            and verification.get("target_runtime_bundle_sha256")
            == RUNTIME_INPUT_SCOPE_TARGET_BUNDLE_SHA256
            and verification.get("runtime_contract_version")
            == RUNTIME_INPUT_SCOPE_CONTRACT_VERSION
            and verification.get("source_release_commit")
            == RUNTIME_INPUT_SCOPE_SOURCE_COMMIT
            and verification.get("target_recipe_version")
            == RUNTIME_INPUT_SCOPE_TARGET_RECIPE_VERSION
            and verification.get("source_artifact_inventories_sha256")
            == RUNTIME_INPUT_SCOPE_SOURCE_ARTIFACT_INVENTORIES_SHA256
            and verification.get("target_change_codes")
            == list(RUNTIME_INPUT_SCOPE_TARGET_CHANGE_CODES)
            and source_backtest_ids == set(RUNTIME_INPUT_SCOPE_SOURCE_BACKTEST_IDS)
        )
    if contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V6:
        return (
            verification.get("repair_generation")
            == FILL_AWARE_HOLDING_AGE_REPAIR_GENERATION
            and verification.get("source_batch_sha256")
            == FILL_AWARE_HOLDING_AGE_SOURCE_BATCH_SHA256
            and verification.get("source_dataset_identity_sha256")
            == FILL_AWARE_HOLDING_AGE_SOURCE_DATASET_IDENTITY_SHA256
            and verification.get("source_dataset_lineage_id")
            == FILL_AWARE_HOLDING_AGE_SOURCE_DATASET_LINEAGE_ID
            and verification.get("source_runner_sha256")
            == FILL_AWARE_HOLDING_AGE_SOURCE_RUNNER_SHA256
            and verification.get(TRANSPARENT_BASELINE_RUNNER_FIELD)
            == FILL_AWARE_HOLDING_AGE_TARGET_RUNNER_SHA256
            and verification.get("source_runtime_bundle_sha256")
            == FILL_AWARE_HOLDING_AGE_SOURCE_BUNDLE_SHA256
            and verification.get("target_runtime_bundle_sha256")
            == FILL_AWARE_HOLDING_AGE_TARGET_BUNDLE_SHA256
            and verification.get("runtime_contract_version")
            == FILL_AWARE_HOLDING_AGE_RUNTIME_CONTRACT_VERSION
            and verification.get("source_release_commit")
            == FILL_AWARE_HOLDING_AGE_SOURCE_COMMIT
            and verification.get("target_recipe_version")
            == FILL_AWARE_HOLDING_AGE_TARGET_RECIPE_VERSION
            and verification.get("source_artifact_inventories_sha256")
            == FILL_AWARE_HOLDING_AGE_SOURCE_ARTIFACT_INVENTORIES_SHA256
            and verification.get("source_unopened_history_selection_sha256")
            == FILL_AWARE_HOLDING_AGE_SOURCE_SELECTION_SHA256
            and verification.get("source_unavailable_horizons_sha256")
            == FILL_AWARE_HOLDING_AGE_SOURCE_UNAVAILABLE_HORIZONS_SHA256
            and verification.get("source_unavailable_evidence_sha256s")
            == sorted(FILL_AWARE_HOLDING_AGE_UNAVAILABLE_EVIDENCE_SHA256S)
            and verification.get("target_change_codes")
            == list(FILL_AWARE_HOLDING_AGE_TARGET_CHANGE_CODES)
            and verification.get("source_bindings")
            == [
                {
                    "backtest_id": backtest_id,
                    "job_id": binding["job_id"],
                    "strategy_version_id": binding["strategy_version_id"],
                }
                for backtest_id, binding in sorted(
                    FILL_AWARE_HOLDING_AGE_SOURCE_BINDINGS.items()
                )
            ]
            and source_backtest_ids
            == set(FILL_AWARE_HOLDING_AGE_SOURCE_BACKTEST_IDS)
        )
    if contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V7:
        return (
            verification.get("repair_generation")
            == DISCRETE_MAX_POSITION_REPAIR_GENERATION
            and verification.get("source_batch_sha256")
            == DISCRETE_MAX_POSITION_SOURCE_BATCH_SHA256
            and verification.get("source_dataset_identity_sha256")
            == DISCRETE_MAX_POSITION_SOURCE_DATASET_IDENTITY_SHA256
            and verification.get("source_dataset_lineage_id")
            == DISCRETE_MAX_POSITION_SOURCE_DATASET_LINEAGE_ID
            and verification.get("source_runner_sha256")
            == DISCRETE_MAX_POSITION_SOURCE_RUNNER_SHA256
            and verification.get(TRANSPARENT_BASELINE_RUNNER_FIELD)
            == DISCRETE_MAX_POSITION_TARGET_RUNNER_SHA256
            and verification.get("source_runtime_bundle_sha256")
            == DISCRETE_MAX_POSITION_SOURCE_BUNDLE_SHA256
            and verification.get("target_runtime_bundle_sha256")
            == DISCRETE_MAX_POSITION_TARGET_BUNDLE_SHA256
            and verification.get("runtime_contract_version")
            == DISCRETE_MAX_POSITION_RUNTIME_CONTRACT_VERSION
            and verification.get("source_release_commit")
            == DISCRETE_MAX_POSITION_SOURCE_COMMIT
            and verification.get("target_recipe_version")
            == DISCRETE_MAX_POSITION_TARGET_RECIPE_VERSION
            and verification.get("source_artifact_inventories_sha256")
            == DISCRETE_MAX_POSITION_SOURCE_ARTIFACT_INVENTORIES_SHA256
            and verification.get("source_unopened_history_selection_sha256")
            == DISCRETE_MAX_POSITION_SOURCE_SELECTION_SHA256
            and verification.get("source_unavailable_horizons_sha256")
            == DISCRETE_MAX_POSITION_SOURCE_UNAVAILABLE_HORIZONS_SHA256
            and verification.get("source_unavailable_evidence_sha256s")
            == sorted(DISCRETE_MAX_POSITION_UNAVAILABLE_EVIDENCE_SHA256S)
            and verification.get("target_change_codes")
            == list(DISCRETE_MAX_POSITION_TARGET_CHANGE_CODES)
            and verification.get("source_bindings")
            == [
                {
                    "backtest_id": backtest_id,
                    "job_id": binding["job_id"],
                    "strategy_version_id": binding["strategy_version_id"],
                }
                for backtest_id, binding in sorted(
                    DISCRETE_MAX_POSITION_SOURCE_BINDINGS.items()
                )
            ]
            and source_backtest_ids
            == set(DISCRETE_MAX_POSITION_SOURCE_BACKTEST_IDS)
        )
    if contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V8:
        return (
            verification.get("repair_generation")
            == TOPK_INDUSTRY_CAPACITY_REPAIR_GENERATION
            and verification.get("source_batch_sha256")
            == TOPK_INDUSTRY_CAPACITY_SOURCE_BATCH_SHA256
            and verification.get("source_dataset_identity_sha256")
            == TOPK_INDUSTRY_CAPACITY_SOURCE_DATASET_IDENTITY_SHA256
            and verification.get("source_dataset_lineage_id")
            == TOPK_INDUSTRY_CAPACITY_SOURCE_DATASET_LINEAGE_ID
            and verification.get("source_runner_sha256")
            == TOPK_INDUSTRY_CAPACITY_SOURCE_RUNNER_SHA256
            and verification.get(TRANSPARENT_BASELINE_RUNNER_FIELD)
            == TOPK_INDUSTRY_CAPACITY_TARGET_RUNNER_SHA256
            and verification.get("source_runtime_bundle_sha256")
            == TOPK_INDUSTRY_CAPACITY_SOURCE_BUNDLE_SHA256
            and verification.get("target_runtime_bundle_sha256")
            == TOPK_INDUSTRY_CAPACITY_TARGET_BUNDLE_SHA256
            and verification.get("runtime_contract_version")
            == TOPK_INDUSTRY_CAPACITY_RUNTIME_CONTRACT_VERSION
            and verification.get("source_release_commit")
            == TOPK_INDUSTRY_CAPACITY_SOURCE_COMMIT
            and verification.get("target_recipe_version")
            == TOPK_INDUSTRY_CAPACITY_TARGET_RECIPE_VERSION
            and verification.get("source_artifact_inventories_sha256")
            == TOPK_INDUSTRY_CAPACITY_SOURCE_ARTIFACT_INVENTORIES_SHA256
            and verification.get("source_unopened_history_selection_sha256")
            == TOPK_INDUSTRY_CAPACITY_SOURCE_SELECTION_SHA256
            and verification.get("source_unavailable_horizons_sha256")
            == TOPK_INDUSTRY_CAPACITY_SOURCE_UNAVAILABLE_HORIZONS_SHA256
            and verification.get("source_unavailable_evidence_sha256s")
            == sorted(TOPK_INDUSTRY_CAPACITY_UNAVAILABLE_EVIDENCE_SHA256S)
            and verification.get("target_change_codes")
            == list(TOPK_INDUSTRY_CAPACITY_TARGET_CHANGE_CODES)
            and verification.get("source_bindings")
            == [
                {
                    "backtest_id": backtest_id,
                    "job_id": binding["job_id"],
                    "strategy_version_id": binding["strategy_version_id"],
                }
                for backtest_id, binding in sorted(
                    TOPK_INDUSTRY_CAPACITY_SOURCE_BINDINGS.items()
                )
            ]
            and source_backtest_ids
            == set(TOPK_INDUSTRY_CAPACITY_SOURCE_BACKTEST_IDS)
        )
    return False


def validate_discrete_max_position_predecessor_registry(
    repair: Any,
    *,
    source_version_id: str,
) -> dict[str, Any]:
    """Require the exact v6 registry row that created the failed v15 source."""

    failure = "discrete max-position predecessor repair registry changed"
    verification_raw = _row_field(repair, "verification_json")
    if not isinstance(verification_raw, Mapping):
        raise ValueError(failure)
    verification = dict(verification_raw)
    source_ids = {
        str(value)
        for value in list(_row_field(repair, "source_backtest_ids_json") or [])
    }
    target_ids = [
        str(value)
        for value in list(
            _row_field(repair, "target_strategy_version_ids_json") or []
        )
    ]
    if (
        str(_row_field(repair, "receipt_sha256") or "")
        != FILL_AWARE_HOLDING_AGE_RECEIPT_SHA256
        or verification.get("receipt_sha256")
        != FILL_AWARE_HOLDING_AGE_RECEIPT_SHA256
        or str(_row_field(repair, "source_batch_sha256") or "")
        != FILL_AWARE_HOLDING_AGE_SOURCE_BATCH_SHA256
        or str(_row_field(repair, "target_batch_sha256") or "")
        != DISCRETE_MAX_POSITION_SOURCE_BATCH_SHA256
        or str(_row_field(repair, "source_dataset_lineage_id") or "")
        != FILL_AWARE_HOLDING_AGE_SOURCE_DATASET_LINEAGE_ID
        or str(_row_field(repair, "target_dataset_lineage_id") or "")
        != DISCRETE_MAX_POSITION_SOURCE_DATASET_LINEAGE_ID
        or str(_row_field(repair, "target_recipe_version") or "")
        != FILL_AWARE_HOLDING_AGE_TARGET_RECIPE_VERSION
        or source_ids != set(FILL_AWARE_HOLDING_AGE_SOURCE_BACKTEST_IDS)
        or target_ids != [str(source_version_id)]
        or not _exact_same_lineage_registry_profile(verification, source_ids)
    ):
        raise ValueError(failure)
    return verification


def validate_topk_industry_capacity_predecessor_registry(
    repair: Any,
    *,
    source_version_id: str,
    audit_event: Any,
) -> dict[str, Any]:
    """Require the exact v7 registry row that created the failed v16 source."""

    failure = "topk industry-capacity predecessor repair registry changed"
    try:
        verification = validate_repair_registry_binding(
            repair,
            lockbox_batch_sha256=TOPK_INDUSTRY_CAPACITY_SOURCE_BATCH_SHA256,
            strategy_version_id=str(source_version_id),
            batch_strategy_version_ids={str(source_version_id)},
            batch_dataset_identity_sha256s={
                TOPK_INDUSTRY_CAPACITY_SOURCE_DATASET_IDENTITY_SHA256
            },
            dataset=TOPK_INDUSTRY_CAPACITY_SOURCE_DATASET,
            dataset_lineage_id=TOPK_INDUSTRY_CAPACITY_SOURCE_DATASET_LINEAGE_ID,
        )
        receipt = validate_pre_result_repair_audit_event(
            audit_event,
            expected_receipt_sha256=DISCRETE_MAX_POSITION_RECEIPT_SHA256,
        )
        created_at = _row_field(audit_event, "created_at")
        if (
            receipt["contract_version"] != PRE_RESULT_REPAIR_CONTRACT_VERSION_V7
            or int(_row_field(audit_event, "id"))
            != int(_row_field(repair, "source_audit_event_id"))
            or verification.get("receipt_created_at") != created_at.isoformat()
        ):
            raise ValueError(failure)
    except (TypeError, ValueError):
        raise ValueError(failure) from None
    return verification


def validate_repair_registry_binding(
    repair: Any,
    *,
    lockbox_batch_sha256: str,
    strategy_version_id: str,
    batch_strategy_version_ids: set[str],
    batch_dataset_identity_sha256s: set[str],
    dataset: str,
    dataset_lineage_id: str,
) -> dict[str, Any]:
    """Validate an exact append-only same-lineage repair registry row.

    This is the shared consumption guard for StrategyStore.  It deliberately
    recognizes only the production-specific same-lineage generations and
    checks the row, JSON verification, complete three-member lockbox and the
    current member in one pure validation step.
    """

    failure = (
        "transparent baseline repair scope has no exact append-only "
        "pre-result registry binding"
    )
    try:
        verification_raw = _row_field(repair, "verification_json")
        if not isinstance(verification_raw, Mapping):
            raise ValueError(failure)
        verification = dict(verification_raw)
        recorded_version_ids = [
            str(value)
            for value in list(
                _row_field(repair, "target_strategy_version_ids_json") or []
            )
        ]
        verification_version_ids = [
            str(value)
            for value in list(
                verification.get("target_strategy_version_ids") or []
            )
        ]
        row_source_ids = {
            str(value)
            for value in list(_row_field(repair, "source_backtest_ids_json") or [])
        }
        verification_source_ids = {
            str(value)
            for value in list(verification.get("source_backtest_ids") or [])
        }
        normalized_versions = {str(value) for value in batch_strategy_version_ids}
        normalized_lineage = _require_sha256(
            dataset_lineage_id, field="dataset_lineage_id"
        )
        normalized_batch = _require_sha256(
            lockbox_batch_sha256, field="lockbox_batch_sha256"
        )
        normalized_identities = {
            _require_sha256(value, field="dataset_identity_sha256")
            for value in batch_dataset_identity_sha256s
        }
        source_audit_event_id = int(_row_field(repair, "source_audit_event_id"))
        row_receipt_sha256 = _require_sha256(
            _row_field(repair, "receipt_sha256"), field="receipt_sha256"
        )
        row_source_batch_sha256 = _require_sha256(
            _row_field(repair, "source_batch_sha256"),
            field="source_batch_sha256",
        )
        row_target_batch_sha256 = _require_sha256(
            _row_field(repair, "target_batch_sha256"),
            field="target_batch_sha256",
        )
        verification_receipt_sha256 = _require_sha256(
            verification.get("receipt_sha256"), field="receipt_sha256"
        )
        verification_source_batch_sha256 = _require_sha256(
            verification.get("source_batch_sha256"),
            field="source_batch_sha256",
        )
        verification_target_batch_sha256 = _require_sha256(
            verification.get("target_batch_sha256"),
            field="target_batch_sha256",
        )
        receipt_created_at = datetime.fromisoformat(
            str(verification.get("receipt_created_at") or "")
        )
        failed_without_results = {
            str(value)
            for value in list(verification.get("failed_without_results") or [])
        }
        results_created_after = list(
            verification.get("results_created_after_preregistration") or []
        )
        is_single_member_repair = verification.get("receipt_contract_version") in {
            PRE_RESULT_REPAIR_CONTRACT_VERSION_V6,
            PRE_RESULT_REPAIR_CONTRACT_VERSION_V7,
            PRE_RESULT_REPAIR_CONTRACT_VERSION_V8,
        }
        expected_member_count = 1 if is_single_member_repair else 3
        if (
            len(normalized_versions) != expected_member_count
            or "" in normalized_versions
            or len(recorded_version_ids) != expected_member_count
            or len(verification_version_ids) != expected_member_count
            or normalized_versions != set(recorded_version_ids)
            or normalized_versions != set(verification_version_ids)
            or str(strategy_version_id) not in normalized_versions
            or len(normalized_identities) != 1
            or row_source_ids != verification_source_ids
            or str(_row_field(repair, "target_dataset_lineage_id"))
            != normalized_lineage
            or str(_row_field(repair, "source_dataset_lineage_id"))
            != normalized_lineage
            or str(verification.get("source_dataset_lineage_id") or "")
            != normalized_lineage
            or str(verification.get("target_dataset_lineage_id") or "")
            != normalized_lineage
            or str(verification.get("target_dataset") or "") != str(dataset)
            or normalized_identities
            != {str(verification.get("target_dataset_identity_sha256") or "")}
            or row_receipt_sha256 != verification_receipt_sha256
            or source_audit_event_id
            != int(verification.get("source_audit_event_id") or -1)
            or row_source_batch_sha256 != verification_source_batch_sha256
            or row_target_batch_sha256 != normalized_batch
            or normalized_batch != verification_target_batch_sha256
            or str(_row_field(repair, "target_recipe_version"))
            != str(verification.get("target_recipe_version") or "")
            or verification.get("contract_version")
            != PRE_RESULT_REPAIR_REGISTRY_VERSION
            or verification.get("performance_information_used") is not False
            or receipt_created_at.tzinfo is None
            or failed_without_results != verification_source_ids
            or results_created_after
            or not _exact_same_lineage_registry_profile(
                verification, verification_source_ids
            )
        ):
            raise ValueError(failure)
    except (TypeError, ValueError):
        raise ValueError(failure) from None
    return verification


class TransparentBaselineLockboxStore:
    """Atomically reserve/recover every statistically available baseline OOS."""

    def __init__(self, database_url: str) -> None:
        self.engine = open_database(database_url)

    def resolve_preregistered_single_member_repair(
        self,
        *,
        calendar_days: Sequence[Any],
        current_recipe_version: str,
    ) -> dict[str, Any] | None:
        """Return one exact allowlisted short repair selection, if registered.

        This lookup never reads performance.  It accepts only the production
        The selected profile independently rechecks the still-failed source,
        partial lockbox, unavailable horizons, artifacts and frozen history.
        """

        current = str(current_recipe_version or "").strip()
        try:
            profile = _single_member_repair_profile_for_target(current)
        except ValueError:
            return None
        calendar = _ordered_calendar(calendar_days)
        matches: list[tuple[Any, dict[str, Any]]] = []
        with self.engine.connect() as connection:
            for audit_row in connection.execute(
                select(audit_events)
                .where(audit_events.c.action == PRE_RESULT_REPAIR_ACTION)
                .order_by(audit_events.c.created_at)
            ).all():
                try:
                    receipt = validate_pre_result_repair_receipt(
                        dict(audit_row.details_json or {})
                    )
                except ValueError:
                    continue
                if receipt["contract_version"] == profile["contract_version"]:
                    matches.append((audit_row, receipt))
            if not matches:
                if current == DISCRETE_MAX_POSITION_TARGET_RECIPE_VERSION:
                    raise ValueError(
                        "v16 discrete max-position repair receipt is not registered"
                    )
                if current == TOPK_INDUSTRY_CAPACITY_TARGET_RECIPE_VERSION:
                    raise ValueError(
                        "v17 topk industry-capacity repair receipt is not registered"
                    )
                return None
            if len(matches) != 1:
                raise ValueError(
                    f"{profile['label']} repair has more than one preregistration"
                )
            audit_row, receipt = matches[0]
            if (
                str(audit_row.method) != "INTERNAL"
                or str(audit_row.path) != "transparent-baseline/pre-result-repair"
                or int(audit_row.status_code) != 201
                or audit_row.created_at is None
            ):
                raise ValueError(f"{profile['label']} repair audit envelope is invalid")

            source_rows: list[Any] = []
            for row in connection.execute(select(oos_vintages)).all():
                raw_link = dict(row.sealed_candidate_set_json or {}).get(
                    "transparent_baseline_lockbox"
                )
                if not isinstance(raw_link, Mapping):
                    continue
                try:
                    link = validate_lockbox_link(raw_link)
                except ValueError:
                    continue
                if link["batch_sha256"] == profile["source_batch_sha256"]:
                    source_rows.append(row)
            if len(source_rows) != 1 or source_rows[0].consumed_at is None:
                raise ValueError(f"{profile['label']} source lockbox is incomplete")
            source_row = source_rows[0]
            sealed = dict(source_row.sealed_candidate_set_json or {})
            link = validate_lockbox_link(sealed.get("transparent_baseline_lockbox"))
            source_binding = next(
                iter(dict(profile["source_bindings"]).values())
            )
            source_version_id = _require_identifier(
                sealed.get("strategy_version_id"), field="strategy_version_id"
            )
            if (
                link["batch_sha256"] != profile["source_batch_sha256"]
                or link["recipe_id"] != "short_relative_strength"
                or link["horizon_profile"] != "short_1_5d"
                or source_version_id != source_binding["strategy_version_id"]
                or str(source_row.dataset_identity or "")
                != profile["source_dataset_identity_sha256"]
                or str(source_row.dataset_lineage_id or "")
                != profile["source_dataset_lineage_id"]
                or source_row.test_start.isoformat()
                != source_binding["periods"]["start"]
                or source_row.test_end.isoformat() != source_binding["periods"]["end"]
            ):
                raise ValueError(f"{profile['label']} source OOS binding changed")
            if current == DISCRETE_MAX_POSITION_TARGET_RECIPE_VERSION:
                predecessor_rows = connection.execute(
                    select(transparent_baseline_pre_result_repairs).where(
                        transparent_baseline_pre_result_repairs.c.target_batch_sha256
                        == DISCRETE_MAX_POSITION_SOURCE_BATCH_SHA256
                    )
                ).all()
                if len(predecessor_rows) != 1:
                    raise ValueError(
                        "discrete max-position predecessor repair registry is incomplete"
                    )
                validate_discrete_max_position_predecessor_registry(
                    predecessor_rows[0], source_version_id=source_version_id
                )
            if current == TOPK_INDUSTRY_CAPACITY_TARGET_RECIPE_VERSION:
                predecessor_rows = connection.execute(
                    select(transparent_baseline_pre_result_repairs).where(
                        transparent_baseline_pre_result_repairs.c.target_batch_sha256
                        == TOPK_INDUSTRY_CAPACITY_SOURCE_BATCH_SHA256
                    )
                ).all()
                if len(predecessor_rows) != 1:
                    raise ValueError(
                        "topk industry-capacity predecessor repair registry is incomplete"
                    )
                predecessor = predecessor_rows[0]
                predecessor_audit = connection.execute(
                    select(audit_events).where(
                        audit_events.c.id == predecessor.source_audit_event_id
                    )
                ).one_or_none()
                if predecessor_audit is None:
                    raise ValueError(
                        "topk industry-capacity predecessor repair audit is incomplete"
                    )
                validate_topk_industry_capacity_predecessor_registry(
                    predecessor,
                    source_version_id=source_version_id,
                    audit_event=predecessor_audit,
                )

            version_row = connection.execute(
                select(strategy_versions).where(
                    strategy_versions.c.id == source_version_id
                )
            ).one()
            source_config = dict(version_row.config_json or {})
            source_bootstrap = dict(source_config.get(BOOTSTRAP_CONFIG_KEY) or {})
            source_lockbox = validate_joint_lockbox(
                source_config.get(LOCKBOX_CONFIG_KEY)
            )
            unavailable = list(source_lockbox.get("unavailable_horizons") or [])
            if (
                source_lockbox["contract_version"] != LOCKBOX_CONTRACT_VERSION_V3
                or source_lockbox["batch_sha256"] != profile["source_batch_sha256"]
                or [item["recipe_id"] for item in source_lockbox["members"]]
                != ["short_relative_strength"]
                or {item["recipe_id"] for item in unavailable}
                != {"swing_trend", "long_quality_value"}
                or canonical_sha256(unavailable)
                != profile["source_unavailable_horizons_sha256"]
                or {item["evidence_sha256"] for item in unavailable}
                != profile["unavailable_evidence_sha256s"]
                or source_config.get("recipe_id") != "short_relative_strength"
                or source_config.get("recipe_version")
                != profile["source_recipe_version"]
                or source_bootstrap.get(TRANSPARENT_BASELINE_RUNNER_FIELD)
                != profile["source_runner_sha256"]
                or source_bootstrap.get(TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD)
                != profile["source_bundle_sha256"]
                or source_bootstrap.get("dataset")
                != profile["source_dataset"]
                or source_bootstrap.get("dataset_identity_sha256")
                != profile["source_dataset_identity_sha256"]
                or source_bootstrap.get("dataset_lineage_id")
                != profile["source_dataset_lineage_id"]
                or dict(source_bootstrap.get("formal_periods") or {})
                != source_binding["periods"]
            ):
                raise ValueError(f"{profile['label']} source contract changed")

            backtest_id = next(iter(dict(profile["source_bindings"])))
            backtest = connection.execute(
                select(
                    backtest_runs,
                    jobs.c.kind.label("job_kind"),
                    jobs.c.status.label("job_status"),
                    jobs.c.error.label("job_error"),
                    jobs.c.payload_json.label("job_payload_json"),
                )
                .join(jobs, jobs.c.id == backtest_runs.c.job_id)
                .where(backtest_runs.c.id == backtest_id)
            ).one()
            member = receipt["members"][0]
            observed_files = _artifact_inventory(backtest.artifact_path)
            source_worker_image = _require_worker_image_digest(
                source_bootstrap.get(TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD),
                field="source worker runtime image digest",
            )
            job_payload = dict(backtest.job_payload_json or {})
            if (
                str(backtest.strategy_version_id) != source_version_id
                or str(backtest.job_id or "") != source_binding["job_id"]
                or str(backtest.dataset or "") != profile["source_dataset"]
                or dict(backtest.periods_json or {}) != source_binding["periods"]
                or str(backtest.status) != "failed"
                or str(backtest.job_kind) != "strategy_backtest"
                or str(backtest.job_status) != "failed"
                or str(backtest.error or "") != source_binding["error"]
                or str(backtest.job_error or "") != source_binding["error"]
                or job_payload.get(
                    TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD
                )
                != source_worker_image
                or backtest.metrics_json is not None
                or _artifact_result_exists(backtest.artifact_path)
                or backtest.created_at is None
                or backtest.created_at > audit_row.created_at
                or observed_files != member["files"]
                or canonical_sha256(observed_files)
                != source_binding["artifact_inventory_sha256"]
            ):
                raise ValueError(f"{profile['label']} source failure evidence changed")

            source_selection = validate_unopened_history_selection(
                source_bootstrap.get("unopened_history_selection"),
                calendar_days=calendar,
            )
            source_batch_evidence = _normalize_prior_batch(
                {
                    "batch_sha256": profile["source_batch_sha256"],
                    "recipe_version": profile["source_recipe_version"],
                    "earliest_final_oos_start": source_row.test_start.isoformat(),
                    "latest_final_oos_end": source_row.test_end.isoformat(),
                    "members": [
                        {
                            "oos_vintage_id": str(source_row.id),
                            "strategy_version_id": source_version_id,
                            "recipe_id": "short_relative_strength",
                            "horizon_profile": "short_1_5d",
                            "recipe_version": profile["source_recipe_version"],
                            "test_start": source_row.test_start.isoformat(),
                            "test_end": source_row.test_end.isoformat(),
                            "first_opened_at": source_row.first_opened_at.isoformat(),
                            "sealed_member_set_sha256": str(
                                source_row.sealed_candidate_set_sha256
                            ),
                        }
                    ],
                    "members_sha256": canonical_sha256(
                        [
                            {
                                "oos_vintage_id": str(source_row.id),
                                "strategy_version_id": source_version_id,
                                "recipe_id": "short_relative_strength",
                                "horizon_profile": "short_1_5d",
                                "recipe_version": profile["source_recipe_version"],
                                "test_start": source_row.test_start.isoformat(),
                                "test_end": source_row.test_end.isoformat(),
                                "first_opened_at": source_row.first_opened_at.isoformat(),
                                "sealed_member_set_sha256": str(
                                    source_row.sealed_candidate_set_sha256
                                ),
                            }
                        ]
                    ),
                }
            )
        evidence = build_pre_result_repair_history_selection(
            calendar_days=calendar,
            current_recipe_version=current,
            source_selection=source_selection,
            repaired_source_batch=source_batch_evidence,
            repair_receipt=receipt,
        )
        selected = calendar[: int(evidence["selected_calendar_trading_days"])]
        return {
            "calendar": selected,
            "evidence": evidence,
            "repair_receipt": receipt,
            "source_batch_sha256": profile["source_batch_sha256"],
            "source_lockbox": source_lockbox,
        }

    def resolve_unopened_history_selection(
        self,
        *,
        calendar_days: Sequence[Any],
        current_recipe_version: str,
        anchored_selection: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Select a stable calendar prefix without reading prior performance.

        Current-recipe rows are deliberately excluded, so reserving v12 and
        reconciling it again cannot move its own cutoff. A partially created
        v12 batch may additionally supply its immutable bootstrap selection as
        an anchor.
        """

        calendar = _ordered_calendar(calendar_days)
        current = str(current_recipe_version or "").strip()
        if anchored_selection is not None:
            evidence = validate_unopened_history_selection(
                anchored_selection,
                calendar_days=calendar,
            )
            if evidence["current_recipe_version"] != current:
                raise ValueError(
                    "transparent baseline anchored history belongs to another recipe"
                )
            selected = calendar[: int(evidence["selected_calendar_trading_days"])]
            return {"calendar": selected, "evidence": evidence}

        with self.engine.connect() as connection:
            rows = connection.execute(select(oos_vintages)).all()
            transparent_rows: list[tuple[Any, dict[str, Any], str]] = []
            version_ids: set[str] = set()
            for row in rows:
                sealed = dict(row.sealed_candidate_set_json or {})
                raw_link = sealed.get("transparent_baseline_lockbox")
                if not isinstance(raw_link, Mapping):
                    continue
                if str(raw_link.get("recipe_id") or "") not in _RECIPE_HORIZONS:
                    continue
                link = validate_lockbox_link(raw_link)
                version_id = _require_identifier(
                    sealed.get("strategy_version_id"),
                    field="strategy_version_id",
                )
                transparent_rows.append((row, link, version_id))
                version_ids.add(version_id)
            version_configs = {
                str(row.id): dict(row.config_json or {})
                for row in (
                    connection.execute(
                        select(
                            strategy_versions.c.id,
                            strategy_versions.c.config_json,
                        ).where(strategy_versions.c.id.in_(version_ids))
                    ).all()
                    if version_ids
                    else []
                )
            }
        if set(version_configs) != version_ids:
            raise ValueError(
                "transparent baseline prior OOS lost its strategy version evidence"
            )
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row, link, version_id in transparent_rows:
            config = version_configs[version_id]
            recipe_id = str(config.get("recipe_id") or "").strip()
            recipe_version = str(config.get("recipe_version") or "").strip()
            if (
                recipe_id != link["recipe_id"]
                or str(config.get("horizon_profile") or "")
                != link["horizon_profile"]
                or not recipe_version
            ):
                raise ValueError(
                    "transparent baseline prior OOS differs from its strategy version"
                )
            grouped.setdefault(str(link["batch_sha256"]), []).append(
                {
                    "oos_vintage_id": str(row.id),
                    "strategy_version_id": version_id,
                    "recipe_id": recipe_id,
                    "horizon_profile": str(link["horizon_profile"]),
                    "recipe_version": recipe_version,
                    "test_start": row.test_start.isoformat(),
                    "test_end": row.test_end.isoformat(),
                    "first_opened_at": row.first_opened_at.isoformat(),
                    "sealed_member_set_sha256": str(
                        row.sealed_candidate_set_sha256
                    ),
                }
            )
        prior_batches: list[dict[str, Any]] = []
        for batch_sha256, members in grouped.items():
            recipe_versions = {str(item["recipe_version"]) for item in members}
            if len(recipe_versions) != 1:
                raise ValueError("transparent baseline OOS batch mixes recipe versions")
            recipe_version = next(iter(recipe_versions))
            if recipe_version == current:
                continue
            ordered_members = sorted(members, key=lambda item: item["recipe_id"])
            prior_batches.append(
                _normalize_prior_batch(
                    {
                        "batch_sha256": batch_sha256,
                        "recipe_version": recipe_version,
                        "earliest_final_oos_start": min(
                            item["test_start"] for item in ordered_members
                        ),
                        "latest_final_oos_end": max(
                            item["test_end"] for item in ordered_members
                        ),
                        "members": ordered_members,
                        "members_sha256": canonical_sha256(ordered_members),
                    }
                )
            )
        evidence = build_unopened_history_selection(
            calendar_days=calendar,
            current_recipe_version=current,
            prior_batches=prior_batches,
        )
        selected = calendar[: int(evidence["selected_calendar_trading_days"])]
        return {"calendar": selected, "evidence": evidence}

    @staticmethod
    def _repair_source_batches(
        rows: Sequence[Any],
        *,
        target_batch_sha256: str,
        superseded_source_batches: set[str] | None = None,
    ) -> dict[str, list[Any]]:
        superseded = superseded_source_batches or set()
        batches: dict[str, list[Any]] = {}
        for row in rows:
            sealed = dict(row.sealed_candidate_set_json or {})
            raw_link = sealed.get("transparent_baseline_lockbox")
            if not isinstance(raw_link, Mapping):
                continue
            try:
                link = validate_lockbox_link(raw_link)
            except ValueError:
                continue
            batch = str(link["batch_sha256"])
            if batch != target_batch_sha256 and batch not in superseded:
                batches.setdefault(batch, []).append(row)
        return batches

    @staticmethod
    def _version_repair_rows(connection: Any, version_ids: set[str]) -> dict[str, Any]:
        rows = connection.execute(
            select(
                strategy_versions.c.id,
                strategy_versions.c.strategy_id,
                strategy_versions.c.horizon_profile,
                strategy_versions.c.benchmark,
                strategy_versions.c.universe,
                strategy_versions.c.config_json,
                strategies.c.economic_hypothesis_group,
            )
            .join(strategies, strategies.c.id == strategy_versions.c.strategy_id)
            .where(strategy_versions.c.id.in_(version_ids))
        ).all()
        if len(rows) != len(version_ids):
            raise ValueError("transparent baseline repair strategy version is missing")
        return {str(row.id): row for row in rows}

    @classmethod
    def _validate_repair_event_binding(
        cls,
        connection: Any,
        *,
        audit_row: Any,
        receipt: Mapping[str, Any],
        source_batch_sha256: str,
        source_rows: Sequence[Any],
        target_batch_sha256: str,
        target_versions: Sequence[Mapping[str, Any]],
        target_expected: Sequence[Mapping[str, Any]],
        target_dataset: str,
        target_dataset_identity_sha256: str,
        target_dataset_lineage_id: str,
    ) -> dict[str, Any]:
        if (
            str(audit_row.method) != "INTERNAL"
            or str(audit_row.path) != "transparent-baseline/pre-result-repair"
            or int(audit_row.status_code) != 201
        ):
            raise ValueError("transparent baseline repair audit envelope is invalid")
        members = {str(item["strategy_version_id"]): item for item in receipt["members"]}
        is_fill_aware_holding_age_repair = receipt["contract_version"] == (
            PRE_RESULT_REPAIR_CONTRACT_VERSION_V6
        )
        is_discrete_max_position_repair = receipt["contract_version"] == (
            PRE_RESULT_REPAIR_CONTRACT_VERSION_V7
        )
        is_topk_industry_capacity_repair = receipt["contract_version"] == (
            PRE_RESULT_REPAIR_CONTRACT_VERSION_V8
        )
        is_single_member_repair = (
            is_fill_aware_holding_age_repair
            or is_discrete_max_position_repair
            or is_topk_industry_capacity_repair
        )
        single_member_profile = (
            _single_member_repair_profile_for_target(
                str(receipt["target_recipe_version"])
            )
            if is_single_member_repair
            else None
        )
        expected_recipe_ids = (
            {"short_relative_strength"}
            if is_single_member_repair
            else set(TRANSPARENT_RESEARCH_BASELINE_IDS)
        )
        expected_member_count = len(expected_recipe_ids)
        source_version_ids = {
            str(dict(row.sealed_candidate_set_json or {}).get("strategy_version_id") or "")
            for row in source_rows
        }
        if (
            len(source_rows) != expected_member_count
            or len(source_version_ids) != expected_member_count
            or set(members) != source_version_ids
            or any(row.consumed_at is None for row in source_rows)
        ):
            raise ValueError("transparent baseline repair source lockbox is incomplete")
        source_links = [
            validate_lockbox_link(
                dict(row.sealed_candidate_set_json or {}).get(
                    "transparent_baseline_lockbox"
                )
            )
            for row in source_rows
        ]
        if (
            {str(item["batch_sha256"]) for item in source_links}
            != {source_batch_sha256}
            or {str(item["recipe_id"]) for item in source_links}
            != expected_recipe_ids
        ):
            raise ValueError("transparent baseline repair source batch binding changed")
        source_lineages = {str(row.dataset_lineage_id or "") for row in source_rows}
        if len(source_lineages) != 1 or "" in source_lineages:
            raise ValueError("transparent baseline repair source lineage is invalid")
        source_lineage = next(iter(source_lineages))
        source_identities = {str(row.dataset_identity or "") for row in source_rows}
        if len(source_identities) != 1 or "" in source_identities:
            raise ValueError("transparent baseline repair source identity is invalid")
        source_identity = next(iter(source_identities))
        is_optimizer_applicability_repair = receipt["contract_version"] == (
            PRE_RESULT_REPAIR_CONTRACT_VERSION_V2
        )
        is_canonical_lf_packaging_repair = receipt["contract_version"] == (
            PRE_RESULT_REPAIR_CONTRACT_VERSION_V3
        )
        is_runtime_alignment_repair = receipt["contract_version"] == (
            PRE_RESULT_REPAIR_CONTRACT_VERSION_V4
        )
        is_runtime_input_scope_repair = receipt["contract_version"] == (
            PRE_RESULT_REPAIR_CONTRACT_VERSION_V5
        )
        is_same_lineage_repair = (
            is_optimizer_applicability_repair
            or is_canonical_lf_packaging_repair
            or is_runtime_alignment_repair
            or is_runtime_input_scope_repair
            or is_single_member_repair
        )
        if is_same_lineage_repair:
            if (
                source_lineage != target_dataset_lineage_id
                or source_identity != target_dataset_identity_sha256
                or any(
                    str(item["dataset"]) != target_dataset
                    for item in receipt["members"]
                )
            ):
                raise ValueError(
                    "same-lineage repair changed the dataset or lineage"
                )
            if is_runtime_input_scope_repair and (
                source_batch_sha256 != RUNTIME_INPUT_SCOPE_SOURCE_BATCH_SHA256
                or source_identity
                != RUNTIME_INPUT_SCOPE_SOURCE_DATASET_IDENTITY_SHA256
                or source_lineage != RUNTIME_INPUT_SCOPE_SOURCE_DATASET_LINEAGE_ID
                or target_dataset_identity_sha256
                != RUNTIME_INPUT_SCOPE_SOURCE_DATASET_IDENTITY_SHA256
                or target_dataset_lineage_id
                != RUNTIME_INPUT_SCOPE_SOURCE_DATASET_LINEAGE_ID
            ):
                raise ValueError("runtime input-scope source batch or dataset changed")
            if is_fill_aware_holding_age_repair and (
                source_batch_sha256 != FILL_AWARE_HOLDING_AGE_SOURCE_BATCH_SHA256
                or source_identity
                != FILL_AWARE_HOLDING_AGE_SOURCE_DATASET_IDENTITY_SHA256
                or source_lineage
                != FILL_AWARE_HOLDING_AGE_SOURCE_DATASET_LINEAGE_ID
                or target_dataset_identity_sha256
                != FILL_AWARE_HOLDING_AGE_SOURCE_DATASET_IDENTITY_SHA256
                or target_dataset_lineage_id
                != FILL_AWARE_HOLDING_AGE_SOURCE_DATASET_LINEAGE_ID
            ):
                raise ValueError("fill-aware holding-age source batch or dataset changed")
            if is_discrete_max_position_repair and (
                source_batch_sha256 != DISCRETE_MAX_POSITION_SOURCE_BATCH_SHA256
                or source_identity
                != DISCRETE_MAX_POSITION_SOURCE_DATASET_IDENTITY_SHA256
                or source_lineage
                != DISCRETE_MAX_POSITION_SOURCE_DATASET_LINEAGE_ID
                or target_dataset_identity_sha256
                != DISCRETE_MAX_POSITION_SOURCE_DATASET_IDENTITY_SHA256
                or target_dataset_lineage_id
                != DISCRETE_MAX_POSITION_SOURCE_DATASET_LINEAGE_ID
            ):
                raise ValueError("discrete max-position source batch or dataset changed")
            if is_topk_industry_capacity_repair and (
                source_batch_sha256 != TOPK_INDUSTRY_CAPACITY_SOURCE_BATCH_SHA256
                or source_identity
                != TOPK_INDUSTRY_CAPACITY_SOURCE_DATASET_IDENTITY_SHA256
                or source_lineage
                != TOPK_INDUSTRY_CAPACITY_SOURCE_DATASET_LINEAGE_ID
                or target_dataset_identity_sha256
                != TOPK_INDUSTRY_CAPACITY_SOURCE_DATASET_IDENTITY_SHA256
                or target_dataset_lineage_id
                != TOPK_INDUSTRY_CAPACITY_SOURCE_DATASET_LINEAGE_ID
            ):
                raise ValueError(
                    "topk industry-capacity source batch or dataset changed"
                )
        elif source_lineage == target_dataset_lineage_id:
            raise ValueError("transparent baseline repair may not fabricate a fresh lineage")

        source_versions = cls._version_repair_rows(connection, source_version_ids)
        target_version_ids = {str(item["id"]) for item in target_versions}
        if len(target_version_ids) != expected_member_count:
            raise ValueError("transparent baseline repair target versions are incomplete")
        target_version_rows = cls._version_repair_rows(connection, target_version_ids)
        target_by_recipe: dict[str, Any] = {}
        for row in target_version_rows.values():
            config = dict(row.config_json or {})
            recipe_id = str(config.get("recipe_id") or "")
            if (
                recipe_id not in expected_recipe_ids
                or config.get("recipe_version") != receipt["target_recipe_version"]
            ):
                raise ValueError("transparent baseline repair target recipe changed")
            target_by_recipe[recipe_id] = row
        if set(target_by_recipe) != expected_recipe_ids:
            raise ValueError("transparent baseline repair target recipes are incomplete")

        source_backtests = connection.execute(
            select(
                backtest_runs,
                jobs.c.kind.label("job_kind"),
                jobs.c.status.label("job_status"),
                jobs.c.error.label("job_error"),
                jobs.c.payload_json.label("job_payload_json"),
            )
            .join(jobs, jobs.c.id == backtest_runs.c.job_id)
            .where(backtest_runs.c.id.in_({str(item["backtest_id"]) for item in members.values()}))
        ).all()
        if len(source_backtests) != expected_member_count:
            raise ValueError("transparent baseline repair source backtests are incomplete")
        by_version = {str(row.strategy_version_id): row for row in source_backtests}
        if set(by_version) != source_version_ids:
            raise ValueError("transparent baseline repair source backtest versions changed")
        receipt_created_at = audit_row.created_at
        if receipt_created_at is None:
            raise ValueError("transparent baseline repair receipt timestamp is missing")
        later_results: list[str] = []
        failed_without_results: list[str] = []
        source_recipe_map = {
            str(link["recipe_id"]): str(
                dict(row.sealed_candidate_set_json or {})["strategy_version_id"]
            )
            for row, link in zip(source_rows, source_links, strict=True)
        }
        expected_by_recipe = {
            str(item["link"]["recipe_id"]): item for item in target_expected
        }
        if set(expected_by_recipe) != expected_recipe_ids:
            raise ValueError("transparent baseline repair target lockbox is incomplete")

        for recipe_id, source_version_id in source_recipe_map.items():
            member = members[source_version_id]
            backtest = by_version[source_version_id]
            source_version = source_versions[source_version_id]
            target_version = target_by_recipe[recipe_id]
            expected = expected_by_recipe[recipe_id]
            recorded_periods = dict(backtest.periods_json or {})
            source_config = dict(source_version.config_json or {})
            target_config = dict(target_version.config_json or {})
            source_bootstrap = dict(source_config.get(BOOTSTRAP_CONFIG_KEY) or {})
            target_bootstrap = dict(target_config.get(BOOTSTRAP_CONFIG_KEY) or {})
            if (
                str(backtest.id) != member["backtest_id"]
                or str(backtest.job_id or "") != member["job_id"]
                or str(backtest.dataset) != member["dataset"]
                or recorded_periods != member["periods"]
                or str(backtest.job_kind) != "strategy_backtest"
                or backtest.created_at is None
                or backtest.created_at > receipt_created_at
                or str(source_version.horizon_profile) != str(expected["link"]["horizon_profile"])
                or str(target_version.horizon_profile) != str(source_version.horizon_profile)
                or str(target_version.strategy_id) != str(source_version.strategy_id)
                or str(target_version.benchmark) != str(source_version.benchmark)
                or str(target_version.universe) != str(source_version.universe)
                or str(target_version.economic_hypothesis_group)
                != str(source_version.economic_hypothesis_group)
                or _repair_economic_config(target_config)
                != _repair_economic_config(source_config)
                or str(expected["test_start"]) != member["periods"]["start"]
                or str(expected["test_end"]) != member["periods"]["end"]
                or (
                    (
                        is_canonical_lf_packaging_repair
                        or is_runtime_alignment_repair
                        or is_runtime_input_scope_repair
                        or is_single_member_repair
                    )
                    and (
                        (
                            not is_single_member_repair
                            and _repair_bootstrap_semantics(target_config)
                            != _repair_bootstrap_semantics(source_config)
                        )
                        or dict(source_bootstrap.get("formal_periods") or {})
                        != member["periods"]
                        or dict(target_bootstrap.get("formal_periods") or {})
                        != member["periods"]
                    )
                )
            ):
                raise ValueError("transparent baseline repair changed an economic or OOS binding")
            if is_optimizer_applicability_repair and source_config.get(
                "recipe_version"
            ) != OPTIMIZER_APPLICABILITY_SOURCE_RECIPE_VERSION:
                raise ValueError("optimizer applicability repair source recipe changed")
            if is_canonical_lf_packaging_repair and source_config.get(
                "recipe_version"
            ) != CANONICAL_LF_PACKAGING_SOURCE_RECIPE_VERSION:
                raise ValueError("canonical LF packaging repair source recipe changed")
            if is_runtime_alignment_repair and source_config.get(
                "recipe_version"
            ) != RUNTIME_ALIGNMENT_SOURCE_RECIPE_VERSION:
                raise ValueError("runtime alignment repair source recipe changed")
            if is_runtime_input_scope_repair and source_config.get(
                "recipe_version"
            ) != RUNTIME_INPUT_SCOPE_SOURCE_RECIPE_VERSION:
                raise ValueError("runtime input-scope repair source recipe changed")
            if is_fill_aware_holding_age_repair and source_config.get(
                "recipe_version"
            ) != FILL_AWARE_HOLDING_AGE_SOURCE_RECIPE_VERSION:
                raise ValueError("fill-aware holding-age repair source recipe changed")
            if is_discrete_max_position_repair and source_config.get(
                "recipe_version"
            ) != DISCRETE_MAX_POSITION_SOURCE_RECIPE_VERSION:
                raise ValueError("discrete max-position repair source recipe changed")
            if is_topk_industry_capacity_repair and source_config.get(
                "recipe_version"
            ) != TOPK_INDUSTRY_CAPACITY_SOURCE_RECIPE_VERSION:
                raise ValueError("topk industry-capacity repair source recipe changed")
            if is_canonical_lf_packaging_repair and (
                source_bootstrap.get(TRANSPARENT_BASELINE_RUNNER_FIELD)
                != receipt["source_runner_expected_sha256"]
            ):
                raise ValueError("canonical LF packaging source runner changed")
            if is_runtime_alignment_repair and (
                source_bootstrap.get(TRANSPARENT_BASELINE_RUNNER_FIELD)
                != receipt["source_runner_sha256"]
                or source_bootstrap.get(TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD)
                is not None
            ):
                raise ValueError("runtime alignment source runtime binding changed")
            if is_runtime_input_scope_repair and (
                source_bootstrap.get(TRANSPARENT_BASELINE_RUNNER_FIELD)
                != receipt["source_runner_sha256"]
                or source_bootstrap.get(TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD)
                != receipt["source_runtime_bundle_sha256"]
            ):
                raise ValueError("runtime input-scope source runtime binding changed")
            if is_fill_aware_holding_age_repair and (
                source_bootstrap.get(TRANSPARENT_BASELINE_RUNNER_FIELD)
                != receipt["source_runner_sha256"]
                or source_bootstrap.get(TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD)
                != receipt["source_runtime_bundle_sha256"]
            ):
                raise ValueError("fill-aware holding-age source runtime binding changed")
            if is_discrete_max_position_repair and (
                source_bootstrap.get(TRANSPARENT_BASELINE_RUNNER_FIELD)
                != receipt["source_runner_sha256"]
                or source_bootstrap.get(TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD)
                != receipt["source_runtime_bundle_sha256"]
            ):
                raise ValueError("discrete max-position source runtime binding changed")
            if is_topk_industry_capacity_repair and (
                source_bootstrap.get(TRANSPARENT_BASELINE_RUNNER_FIELD)
                != receipt["source_runner_sha256"]
                or source_bootstrap.get(TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD)
                != receipt["source_runtime_bundle_sha256"]
            ):
                raise ValueError(
                    "topk industry-capacity source runtime binding changed"
                )
            if is_same_lineage_repair and (
                target_bootstrap.get(TRANSPARENT_BASELINE_RUNNER_FIELD)
                != receipt[TRANSPARENT_BASELINE_RUNNER_FIELD]
            ):
                raise ValueError("same-lineage repair target runner changed")
            if (
                is_runtime_alignment_repair
                or is_runtime_input_scope_repair
                or is_single_member_repair
            ) and (
                target_bootstrap.get(TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD)
                != receipt["target_runtime_bundle_sha256"]
            ):
                raise ValueError("runtime repair target runtime bundle changed")
            if is_single_member_repair:
                assert single_member_profile is not None
                source_lockbox = validate_joint_lockbox(
                    source_config.get(LOCKBOX_CONFIG_KEY)
                )
                target_lockbox = validate_joint_lockbox(
                    target_config.get(LOCKBOX_CONFIG_KEY)
                )
                source_unavailable = list(
                    source_lockbox.get("unavailable_horizons") or []
                )
                target_unavailable = list(
                    target_lockbox.get("unavailable_horizons") or []
                )
                source_selection = validate_unopened_history_selection(
                    source_bootstrap.get("unopened_history_selection")
                )
                target_selection = validate_unopened_history_selection(
                    target_bootstrap.get("unopened_history_selection")
                )
                source_semantics = _repair_bootstrap_semantics(source_config)
                target_semantics = _repair_bootstrap_semantics(target_config)
                source_semantics.pop("unopened_history_selection", None)
                target_semantics.pop("unopened_history_selection", None)
                source_worker_image = _require_worker_image_digest(
                    source_bootstrap.get(
                        TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD
                    ),
                    field="source worker runtime image digest",
                )
                target_worker_image = _require_worker_image_digest(
                    target_bootstrap.get(
                        TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD
                    ),
                    field="target worker runtime image digest",
                )
                source_semantics.pop(
                    TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD,
                    None,
                )
                target_semantics.pop(
                    TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD,
                    None,
                )
                source_job_payload = dict(backtest.job_payload_json or {})
                if (
                    source_semantics != target_semantics
                    or source_job_payload.get(
                        TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD
                    )
                    != source_worker_image
                    or target_worker_image
                    != target_bootstrap.get(
                        TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD
                    )
                    or source_lockbox["contract_version"]
                    != LOCKBOX_CONTRACT_VERSION_V3
                    or target_lockbox["contract_version"]
                    != LOCKBOX_CONTRACT_VERSION_V3
                    or source_lockbox["batch_sha256"]
                    != single_member_profile["source_batch_sha256"]
                    or source_unavailable != target_unavailable
                    or canonical_sha256(source_unavailable)
                    != single_member_profile[
                        "source_unavailable_horizons_sha256"
                    ]
                    or {item["evidence_sha256"] for item in source_unavailable}
                    != single_member_profile["unavailable_evidence_sha256s"]
                    or source_selection["selection_sha256"]
                    != single_member_profile["source_selection_sha256"]
                    or target_selection["contract_version"]
                    != single_member_profile["target_history_contract"]
                    or target_selection["repaired_source_batch_sha256"]
                    != single_member_profile["source_batch_sha256"]
                    or target_selection["repair_receipt_sha256"]
                    != receipt["receipt_sha256"]
                    or target_selection["performance_information_used"] is not False
                ):
                    raise ValueError(
                        f"{single_member_profile['label']} repair changed "
                        "partial-lockbox evidence"
                    )
            has_metrics = backtest.metrics_json is not None
            has_result = _artifact_result_exists(backtest.artifact_path)
            declared_files = sorted(member["files"], key=lambda item: item["path"])
            observed_files = _artifact_inventory(backtest.artifact_path)
            observed_by_path = {item["path"]: item for item in observed_files}
            if any(
                observed_by_path.get(item["path"]) != item for item in declared_files
            ):
                raise ValueError("transparent baseline repair artifact evidence changed")
            if member["status"] == "failed" and observed_files != declared_files:
                raise ValueError("transparent baseline repair failure artifacts changed")
            if has_metrics or has_result:
                if (
                    member["status"] != "running"
                    or backtest.finished_at is None
                    or not receipt_created_at < backtest.finished_at
                ):
                    raise ValueError(
                        "transparent baseline repair was registered after performance existed"
                    )
                later_results.append(str(backtest.id))
            elif member["status"] == "failed":
                marker = str(member.get("error") or "")
                if (
                    is_canonical_lf_packaging_repair
                    or is_runtime_alignment_repair
                    or is_runtime_input_scope_repair
                    or is_single_member_repair
                ):
                    error_matches = (
                        str(backtest.error or "") == marker
                        and str(backtest.job_error or "") == marker
                    )
                else:
                    error_matches = (
                        marker in str(backtest.error or "")
                        and marker in str(backtest.job_error or "")
                    )
                if (
                    str(backtest.status) != "failed"
                    or str(backtest.job_status) != "failed"
                    or not error_matches
                ):
                    raise ValueError("transparent baseline repair failure evidence changed")
                failed_without_results.append(str(backtest.id))
            else:
                # The running member was preregistered without performance. It
                # may still be running/cancelled/failed, but if it eventually
                # produced a result the branch above requires a later finish.
                if backtest.finished_at is not None and backtest.finished_at <= receipt_created_at:
                    raise ValueError(
                        "transparent baseline running member ended before preregistration"
                    )

        if not is_same_lineage_repair and (
            str(target_dataset or "").strip()
            == str(receipt["members"][0]["dataset"])
        ):
            raise ValueError("transparent baseline repair target dataset was not rematerialized")
        return {
            "contract_version": PRE_RESULT_REPAIR_REGISTRY_VERSION,
            "receipt_sha256": str(receipt["receipt_sha256"]),
            "source_audit_event_id": int(audit_row.id),
            "source_batch_sha256": source_batch_sha256,
            "target_batch_sha256": target_batch_sha256,
            "source_dataset_lineage_id": source_lineage,
            **(
                {
                    "source_dataset_identity_sha256": receipt[
                        "source_dataset_identity_sha256"
                    ]
                }
                if is_runtime_input_scope_repair or is_single_member_repair
                else {}
            ),
            "target_dataset": target_dataset,
            "target_dataset_identity_sha256": target_dataset_identity_sha256,
            "target_dataset_lineage_id": target_dataset_lineage_id,
            "target_recipe_version": str(receipt["target_recipe_version"]),
            "receipt_contract_version": str(receipt["contract_version"]),
            "repair_generation": receipt.get("repair_generation"),
            "source_release_commit": receipt.get("source_release_commit"),
            "source_runner_expected_sha256": receipt.get(
                "source_runner_expected_sha256"
            ),
            "source_runner_observed_sha256": receipt.get(
                "source_runner_observed_sha256"
            ),
            "source_runner_sha256": receipt.get("source_runner_sha256"),
            "source_runtime_bundle_sha256": receipt.get(
                "source_runtime_bundle_sha256"
            ),
            "target_runtime_bundle_sha256": receipt.get(
                "target_runtime_bundle_sha256"
            ),
            TRANSPARENT_BASELINE_RUNNER_FIELD: receipt.get(
                TRANSPARENT_BASELINE_RUNNER_FIELD
            ),
            "packaging_contract_version": receipt.get(
                "packaging_contract_version"
            ),
            "runtime_contract_version": receipt.get("runtime_contract_version"),
            "source_artifact_inventories_sha256": receipt.get(
                "source_artifact_inventories_sha256"
            ),
            **(
                {"target_change_codes": list(receipt["target_change_codes"])}
                if is_runtime_input_scope_repair or is_single_member_repair
                else {}
            ),
            **(
                {
                    "source_unopened_history_selection_sha256": receipt[
                        "source_unopened_history_selection_sha256"
                    ],
                    "source_unavailable_horizons_sha256": receipt[
                        "source_unavailable_horizons_sha256"
                    ],
                    "source_unavailable_evidence_sha256s": list(
                        receipt["source_unavailable_evidence_sha256s"]
                    ),
                    "source_bindings": [
                        {
                            "backtest_id": item["backtest_id"],
                            "job_id": item["job_id"],
                            "strategy_version_id": item["strategy_version_id"],
                        }
                        for item in receipt["members"]
                    ],
                }
                if is_single_member_repair
                else {}
            ),
            "source_backtest_ids": sorted(item["backtest_id"] for item in members.values()),
            "target_strategy_version_ids": sorted(target_version_ids),
            "receipt_created_at": receipt_created_at.isoformat(),
            "failed_without_results": sorted(failed_without_results),
            "results_created_after_preregistration": sorted(later_results),
            "performance_information_used": False,
        }

    @classmethod
    def _register_pre_result_repair(
        cls,
        connection: Any,
        *,
        source_batch_sha256: str,
        source_rows: Sequence[Any],
        target_batch_sha256: str,
        target_versions: Sequence[Mapping[str, Any]],
        target_expected: Sequence[Mapping[str, Any]],
        target_dataset: str,
        target_dataset_identity_sha256: str,
        target_dataset_lineage_id: str,
    ) -> dict[str, Any]:
        existing = connection.execute(
            select(transparent_baseline_pre_result_repairs).where(
                transparent_baseline_pre_result_repairs.c.target_batch_sha256
                == target_batch_sha256
            )
        ).first()
        if existing is not None:
            if str(existing.source_batch_sha256) != source_batch_sha256:
                raise ValueError("transparent baseline repair registry target was rebound")
            return dict(existing.verification_json or {})
        matches: list[tuple[Any, dict[str, Any], dict[str, Any]]] = []
        for audit_row in connection.execute(
            select(audit_events)
            .where(audit_events.c.action == PRE_RESULT_REPAIR_ACTION)
            .order_by(audit_events.c.created_at)
            .with_for_update()
        ).all():
            try:
                receipt = validate_pre_result_repair_receipt(
                    dict(audit_row.details_json or {})
                )
                verification = cls._validate_repair_event_binding(
                    connection,
                    audit_row=audit_row,
                    receipt=receipt,
                    source_batch_sha256=source_batch_sha256,
                    source_rows=source_rows,
                    target_batch_sha256=target_batch_sha256,
                    target_versions=target_versions,
                    target_expected=target_expected,
                    target_dataset=target_dataset,
                    target_dataset_identity_sha256=target_dataset_identity_sha256,
                    target_dataset_lineage_id=target_dataset_lineage_id,
                )
            except ValueError:
                continue
            matches.append((audit_row, receipt, verification))
        if len(matches) != 1:
            raise ValueError(
                "transparent baseline final OOS has more than one prior batch or "
                "requires exactly one valid pre-result repair receipt"
            )
        audit_row, receipt, verification = matches[0]
        connection.execute(
            insert(transparent_baseline_pre_result_repairs).values(
                receipt_sha256=receipt["receipt_sha256"],
                source_audit_event_id=int(audit_row.id),
                source_batch_sha256=source_batch_sha256,
                target_batch_sha256=target_batch_sha256,
                source_dataset_lineage_id=verification[
                    "source_dataset_lineage_id"
                ],
                target_dataset_lineage_id=target_dataset_lineage_id,
                target_recipe_version=receipt["target_recipe_version"],
                source_backtest_ids_json=verification["source_backtest_ids"],
                target_strategy_version_ids_json=verification[
                    "target_strategy_version_ids"
                ],
                verification_json=verification,
                created_at=datetime.now(UTC),
            )
        )
        return verification

    def reserve(
        self,
        *,
        versions: Sequence[Mapping[str, Any]],
        dataset: str,
        dataset_identity_sha256: str,
        dataset_lineage_id: str,
    ) -> dict[str, Any]:
        if not 1 <= len(versions) <= 3:
            raise ValueError("baseline lockbox reservation requires one to three versions")
        identity = _require_sha256(
            dataset_identity_sha256,
            field="dataset_identity_sha256",
        )
        lineage = _require_sha256(dataset_lineage_id, field="dataset_lineage_id")
        expected: list[dict[str, Any]] = []
        batch_ids: set[str] = set()
        lockbox_contract_versions: set[str] = set()
        for version in versions:
            config = version.get("config")
            if not isinstance(config, Mapping):
                raise ValueError("joint lockbox strategy config is invalid")
            lockbox = validate_joint_lockbox(config.get(LOCKBOX_CONFIG_KEY))
            if (
                lockbox["dataset"] != dataset
                or lockbox["dataset_identity_sha256"] != identity
                or lockbox["dataset_lineage_id"] != lineage
            ):
                raise ValueError("joint lockbox dataset binding changed")
            batch_ids.add(str(lockbox["batch_sha256"]))
            lockbox_contract_versions.add(str(lockbox["contract_version"]))
            member_set = baseline_oos_sealed_member_set(version)
            link = validate_lockbox_link(member_set["transparent_baseline_lockbox"])
            bootstrap = config.get(BOOTSTRAP_CONFIG_KEY)
            if not isinstance(bootstrap, Mapping):
                raise ValueError("joint lockbox member has no bootstrap contract")
            periods = dict(bootstrap.get("formal_periods") or {})
            expected.append(
                {
                    "version_id": str(version["id"]),
                    "test_start": date.fromisoformat(str(periods["start"])),
                    "test_end": date.fromisoformat(str(periods["end"])),
                    "sealed_member_set": member_set,
                    "sealed_member_set_sha256": canonical_sha256(member_set),
                    "link": link,
                }
            )
        if len(batch_ids) != 1:
            raise ValueError("public baseline versions do not share one joint lockbox")
        if len(lockbox_contract_versions) != 1:
            raise ValueError("public baseline versions mix lockbox contract versions")
        lockbox_contract_version = next(iter(lockbox_contract_versions))
        if (
            len(versions) != 3
            and lockbox_contract_version != LOCKBOX_CONTRACT_VERSION_V3
        ):
            raise ValueError(
                "partial baseline reservation requires the unavailable-horizons contract"
            )
        observed_links = {item["link"]["member_sha256"] for item in expected}
        declared_links = set(expected[0]["link"]["member_sha256s"])
        if observed_links != declared_links:
            raise ValueError("joint lockbox versions do not cover all declared members")
        target_batch_sha256 = next(iter(batch_ids))
        base_scope = f"lineage:{lineage}"
        scope = base_scope
        earliest = min(item["test_start"] for item in expected)
        latest = max(item["test_end"] for item in expected)
        now = datetime.now(UTC)
        repair_registration: dict[str, Any] | None = None
        with self.engine.begin() as connection:
            connection.execute(
                text("SELECT pg_advisory_xact_lock(hashtext(:oos_scope))"),
                {"oos_scope": base_scope},
            )
            # A changed builder/eligibility contract produces a real new
            # lineage, but lineage churn is not permission to look at the same
            # final-OOS window twice. Scan all prior transparent lockboxes.
            # The only exception is a hashed receipt recorded before any
            # performance result and consumed into the append-only registry in
            # this same transaction.
            all_overlapping_rows = connection.execute(
                select(oos_vintages)
                .where(
                    oos_vintages.c.test_start <= latest,
                    oos_vintages.c.test_end >= earliest,
                )
                .with_for_update()
            ).all()
            registered_repairs = connection.execute(
                select(transparent_baseline_pre_result_repairs)
            ).all()
            superseded_source_batches = {
                str(row.source_batch_sha256) for row in registered_repairs
            }
            existing_target_repairs = [
                row
                for row in registered_repairs
                if str(row.target_batch_sha256) == target_batch_sha256
            ]
            if len(existing_target_repairs) > 1:
                raise ValueError("transparent baseline repair target is duplicated")
            if existing_target_repairs:
                repair_registration = dict(
                    existing_target_repairs[0].verification_json or {}
                )
            source_batches = self._repair_source_batches(
                all_overlapping_rows,
                target_batch_sha256=target_batch_sha256,
                superseded_source_batches=superseded_source_batches,
            )
            if source_batches and repair_registration is None:
                is_exact_single_member_repair_target = (
                    len(expected) == 1
                    and expected[0]["link"]["recipe_id"]
                    == "short_relative_strength"
                    and str(dict(versions[0].get("config") or {}).get("recipe_version") or "")
                    in {
                        FILL_AWARE_HOLDING_AGE_TARGET_RECIPE_VERSION,
                        DISCRETE_MAX_POSITION_TARGET_RECIPE_VERSION,
                        TOPK_INDUSTRY_CAPACITY_TARGET_RECIPE_VERSION,
                    }
                )
                if len(expected) != 3 and not is_exact_single_member_repair_target:
                    raise ValueError(
                        "partial baseline lockboxes cannot reuse an opened final OOS"
                    )
                if len(source_batches) != 1:
                    raise ValueError(
                        "transparent baseline final OOS already has more than one prior batch"
                    )
                source_batch_sha256, source_rows = next(iter(source_batches.items()))
                repair_registration = self._register_pre_result_repair(
                    connection,
                    source_batch_sha256=source_batch_sha256,
                    source_rows=source_rows,
                    target_batch_sha256=target_batch_sha256,
                    target_versions=versions,
                    target_expected=expected,
                    target_dataset=dataset,
                    target_dataset_identity_sha256=identity,
                    target_dataset_lineage_id=lineage,
                )
            if repair_registration is not None and _exact_same_lineage_registry_profile(
                repair_registration,
                {
                    str(value)
                    for value in list(
                        repair_registration.get("source_backtest_ids") or []
                    )
                },
            ):
                # Allowlisted same-lineage generations intentionally keep
                # the exact data and OOS dates. Use a repair-specific scope so
                # each generation receives new immutable one-shot rows instead
                # of mutating or reopening its consumed source rows.
                scope = f"{base_scope}:repair:{target_batch_sha256}"
            scope_filter = oos_vintages.c.scope == scope
            if scope == base_scope:
                scope_filter = or_(
                    scope_filter,
                    oos_vintages.c.scope.like("dataset:%"),
                )
            rows = connection.execute(
                select(oos_vintages)
                .where(
                    scope_filter,
                    oos_vintages.c.test_start <= latest,
                    oos_vintages.c.test_end >= earliest,
                )
                .with_for_update()
            ).all()
            if rows:
                expected_by_window = {
                    (item["test_start"], item["test_end"]): item for item in expected
                }
                if len(rows) != len(expected):
                    raise ValueError(
                        "joint lockbox overlaps another reserved or consumed OOS vintage"
                    )
                for row in rows:
                    item = expected_by_window.get((row.test_start, row.test_end))
                    if (
                        item is None
                        or str(row.scope) != scope
                        or str(row.dataset_identity) != identity
                        or str(row.dataset_lineage_id or "") != lineage
                        or dict(row.sealed_candidate_set_json or {})
                        != item["sealed_member_set"]
                        or str(row.sealed_candidate_set_sha256)
                        != item["sealed_member_set_sha256"]
                    ):
                        raise ValueError(
                            "existing OOS vintages differ from the joint lockbox"
                        )
            else:
                for item in expected:
                    connection.execute(
                        insert(oos_vintages).values(
                            id=uuid.uuid4().hex,
                            scope=scope,
                            dataset_identity=identity,
                            dataset_lineage_id=lineage,
                            test_start=item["test_start"],
                            test_end=item["test_end"],
                            sealed_at=now,
                            first_opened_at=now,
                            consumed_at=None,
                            capital_oos_alpha_batch_id=None,
                            sealed_candidate_set_json=item["sealed_member_set"],
                            sealed_candidate_set_sha256=item[
                                "sealed_member_set_sha256"
                            ],
                            created_at=now,
                        )
                    )
            recorded = connection.execute(
                select(oos_vintages).where(
                    oos_vintages.c.scope == scope,
                    oos_vintages.c.test_start <= latest,
                    oos_vintages.c.test_end >= earliest,
                )
            ).all()
        members = []
        for row in sorted(recorded, key=lambda item: item.test_start):
            sealed = dict(row.sealed_candidate_set_json or {})
            link = validate_lockbox_link(sealed.get("transparent_baseline_lockbox"))
            members.append(
                {
                    "oos_vintage_id": str(row.id),
                    "strategy_version_id": str(
                        sealed.get("strategy_version_id") or ""
                    ),
                    "recipe_id": link["recipe_id"],
                    "horizon_profile": link["horizon_profile"],
                    "test_start": row.test_start.isoformat(),
                    "test_end": row.test_end.isoformat(),
                    "status": "consumed" if row.consumed_at is not None else "reserved",
                    "consumed_at": row.consumed_at,
                }
            )
        return {
            "contract_version": lockbox_contract_version,
            "batch_sha256": target_batch_sha256,
            "scope": scope,
            "dataset": dataset,
            "dataset_identity_sha256": identity,
            "dataset_lineage_id": lineage,
            "pre_result_repair": repair_registration,
            "members": members,
        }

    def get(self, batch_sha256: str) -> dict[str, Any]:
        batch = _require_sha256(batch_sha256, field="batch_sha256")
        with self.engine.connect() as connection:
            rows = connection.execute(select(oos_vintages)).all()
        matches = []
        for row in rows:
            sealed = dict(row.sealed_candidate_set_json or {})
            raw_link = sealed.get("transparent_baseline_lockbox")
            if not isinstance(raw_link, Mapping) or raw_link.get("batch_sha256") != batch:
                continue
            link = validate_lockbox_link(raw_link)
            matches.append(
                {
                    **row_dict(row),
                    "recipe_id": link["recipe_id"],
                    "horizon_profile": link["horizon_profile"],
                }
            )
        first_link = (
            validate_lockbox_link(
                matches[0]["sealed_candidate_set_json"].get(
                    "transparent_baseline_lockbox"
                )
            )
            if matches
            else None
        )
        expected_count = len(first_link["member_sha256s"]) if first_link else 0
        if not matches or len(matches) != expected_count:
            raise KeyError(batch)
        return {"batch_sha256": batch, "members": matches}
