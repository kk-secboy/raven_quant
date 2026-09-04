"""Immutable runner identities for allowlisted transparent-baseline repairs."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from quant_platform.runtime_source_closure import position_risk_source_closure_sha256

OPTIMIZER_APPLICABILITY_TARGET_RECIPE_VERSION = (
    "qlib-rdagent-single-mainline-2026-08-30-v8"
)
OPTIMIZER_APPLICABILITY_TARGET_RUNNER_SHA256 = (
    "fa1090deaa66ca77a045c1a872f7b6451043e116e717954533217908135c584e"
)
CANONICAL_LF_TARGET_RECIPE_VERSION = "qlib-rdagent-single-mainline-2026-08-30-v9"
CANONICAL_LF_TARGET_RUNNER_SHA256 = (
    "256bbfd579865e7bc1442f1f64003d655241abecb320e5d27b440dd635224d57"
)
RUNTIME_ALIGNMENT_TARGET_RECIPE_VERSION = (
    "qlib-rdagent-single-mainline-2026-08-30-v10"
)
RUNTIME_ALIGNMENT_TARGET_RUNNER_SHA256 = (
    "0f868f2f3beaf5cff5db461f11c411ab3ef960c620c3f2f902ac385fc79eff4d"
)
RUNTIME_ALIGNMENT_RUNTIME_PATHS = (
    "scripts/run_multifactor_backtest.py",
    "scripts/run_recommendation_refresh.py",
    "src/quant_platform/portfolio_policy.py",
    "src/quant_platform/strategy_backtest.py",
    "src/quant_platform/strategy_rule_runtime.py",
)
RUNTIME_ALIGNMENT_SOURCE_RUNTIME_BUNDLE_SHA256 = (
    "b2b501cf1b59201b8732bacfa787ab38e41867993234aa2f9f605d9b163e7c68"
)
RUNTIME_ALIGNMENT_TARGET_RUNTIME_BUNDLE_SHA256 = (
    "eea7ca8854acbed0370ead8d97fdfdb128059f41d8a5995a5050b7ec9e9f3a34"
)
RUNTIME_INPUT_SCOPE_TARGET_RECIPE_VERSION = (
    "qlib-rdagent-single-mainline-2026-08-30-v11"
)
RUNTIME_INPUT_SCOPE_TARGET_RUNNER_SHA256 = (
    "64fa634b4e774279356c5655e70c741890ed9b208ccf75df757a378c0f56a432"
)
RUNTIME_INPUT_SCOPE_TARGET_RUNTIME_BUNDLE_SHA256 = (
    "687ad83efd9d734a238bd6b52b3f5e670cec1e5d165ec2b25cabcef724feb7cf"
)
POSITION_RISK_TARGET_RECIPE_VERSION = (
    "qlib-rdagent-single-mainline-2026-08-30-v12"
)
POSITION_RISK_TARGET_RUNNER_SHA256 = (
    "31d4c7a294ae61c19edcdb8e014b521c1b544836d6891af36cfba3ecf4ab43a3"
)
POSITION_RISK_TARGET_RUNTIME_BUNDLE_SHA256 = (
    "6366bd2b77c1c60ea4069afde43d5335c1e362c18acf2955f13902e7ba9ccbc6"
)
FAIL_CLOSED_EXECUTION_TARGET_RECIPE_VERSION = (
    "qlib-rdagent-single-mainline-2026-08-30-v13"
)
FAIL_CLOSED_EXECUTION_TARGET_RUNNER_SHA256 = (
    "31d4c7a294ae61c19edcdb8e014b521c1b544836d6891af36cfba3ecf4ab43a3"
)
FAIL_CLOSED_EXECUTION_TARGET_RUNTIME_BUNDLE_SHA256 = (
    "3000d84d183fe589da13d01902f88b1da1d402f47e18e4aafe91831cc59bdd8f"
)
FILL_AWARE_HOLDING_AGE_TARGET_RECIPE_VERSION = (
    "qlib-rdagent-single-mainline-2026-08-30-v14"
)
FILL_AWARE_HOLDING_AGE_TARGET_RUNNER_SHA256 = (
    "31d4c7a294ae61c19edcdb8e014b521c1b544836d6891af36cfba3ecf4ab43a3"
)
FILL_AWARE_HOLDING_AGE_TARGET_RUNTIME_BUNDLE_SHA256 = (
    "04f9dce110aaf44d32972c68db3edefafbd08cfdbf1f7dfb98aac9a15409bfe5"
)
SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RECIPE_VERSION = (
    "qlib-rdagent-single-mainline-2026-08-30-v15"
)
SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RUNNER_SHA256 = (
    "31d4c7a294ae61c19edcdb8e014b521c1b544836d6891af36cfba3ecf4ab43a3"
)
# Filled only after the complete v15 source closure is stable in two
# consecutive calculations.  Keeping this assignment in the same normalized
# seal family prevents the digest from hashing itself.
SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RUNTIME_BUNDLE_SHA256 = (
    "e361d85d69d77cb6e0de7072db6f8aaff5d83f1e7902fe16ef06c2e28fce1867"
)
DISCRETE_MAX_POSITION_REPAIR_TARGET_RECIPE_VERSION = (
    "qlib-rdagent-single-mainline-2026-08-30-v16"
)
DISCRETE_MAX_POSITION_REPAIR_TARGET_RUNNER_SHA256 = (
    "31d4c7a294ae61c19edcdb8e014b521c1b544836d6891af36cfba3ecf4ab43a3"
)
DISCRETE_MAX_POSITION_REPAIR_TARGET_RUNTIME_BUNDLE_SHA256 = (
    "ac1c2020996efa4e3c3e9609dd4c736d8c65893dccd7ba9c90e3464402d2a329"
)
TOPK_INDUSTRY_CAPACITY_REPAIR_TARGET_RECIPE_VERSION = (
    "qlib-rdagent-single-mainline-2026-08-31-v17"
)
TOPK_INDUSTRY_CAPACITY_REPAIR_TARGET_RUNNER_SHA256 = (
    "31d4c7a294ae61c19edcdb8e014b521c1b544836d6891af36cfba3ecf4ab43a3"
)
# Filled only after the complete v17 source closure is stable in two
# consecutive calculations. The source-closure normalizer excludes this
# release seal from its own digest.
TOPK_INDUSTRY_CAPACITY_REPAIR_TARGET_RUNTIME_BUNDLE_SHA256 = (
    "c3d6aeb00a8f5286ee117f885231b43ad49e461c5183f5cbf32c4601824bb9fc"
)
FORWARD_ONLY_REHABILITATION_TARGET_RECIPE_VERSION = (
    "qlib-rdagent-single-mainline-2026-08-31-v18"
)
FORWARD_ONLY_REHABILITATION_TARGET_RUNNER_SHA256 = (
    "c45bd901f25ba0cf289e3ec1a71015865d190afb4510c6bc207d53c9cc2ab24d"
)
# Filled after the complete v18 source closure is stable. The source-closure
# normalizer excludes both v18 seal assignments from their own digest.
FORWARD_ONLY_REHABILITATION_TARGET_RUNTIME_BUNDLE_SHA256 = (
    "3f0c60adbe3b50ff26771e11ce75f6a48d13772ac5bff549f716469748b92874"
)
STRATEGY_RESEARCH_V20_TARGET_RECIPE_VERSION = (
    "qlib-rdagent-single-mainline-2026-09-01-v20"
)
STRATEGY_RESEARCH_V20_TARGET_RUNNER_SHA256 = (
    "c45bd901f25ba0cf289e3ec1a71015865d190afb4510c6bc207d53c9cc2ab24d"
)
# Historical v20 seal.  It stays independently addressable after v21 becomes
# current so old evidence cannot be rebound to the new source closure.
STRATEGY_RESEARCH_V20_TARGET_RUNTIME_BUNDLE_SHA256 = (
    "a0fe9491b27836526af9d7644c2c396973a95933fcfa4bf9365046cf634be53d"
)
STRATEGY_RESEARCH_V21_TARGET_RECIPE_VERSION = (
    "qlib-rdagent-single-mainline-2026-09-01-v21"
)
STRATEGY_RESEARCH_V21_TARGET_RUNNER_SHA256 = (
    "c45bd901f25ba0cf289e3ec1a71015865d190afb4510c6bc207d53c9cc2ab24d"
)
# Historical v21 seal. It remains independently addressable after later
# versions become current so existing evidence cannot be rebound.
STRATEGY_RESEARCH_V21_TARGET_RUNTIME_BUNDLE_SHA256 = (
    "5859384d80aeee8302e1474f08b405eaaccec0e2b056b9f1da89e72454e42f48"
)
STRATEGY_RESEARCH_V22_TARGET_RECIPE_VERSION = (
    "qlib-rdagent-single-mainline-2026-09-01-v22"
)
STRATEGY_RESEARCH_V22_TARGET_RUNNER_SHA256 = (
    "c45bd901f25ba0cf289e3ec1a71015865d190afb4510c6bc207d53c9cc2ab24d"
)
# Historical v22 seal. It remains independently addressable after v23 becomes
# current so already-applied database constraints and evidence stay immutable.
STRATEGY_RESEARCH_V22_TARGET_RUNTIME_BUNDLE_SHA256 = (
    "6709b7e2a4abcc2ae9dcb63db2d50f4bf8ba5507ee47c62ee95d70109fe7d7e0"
)
STRATEGY_RESEARCH_V23_TARGET_RECIPE_VERSION = (
    "qlib-rdagent-single-mainline-2026-09-01-v23"
)
STRATEGY_RESEARCH_V23_TARGET_RUNNER_SHA256 = (
    "c45bd901f25ba0cf289e3ec1a71015865d190afb4510c6bc207d53c9cc2ab24d"
)
# Historical v23 seal. It remains independently addressable after v24 becomes
# current so already-applied database constraints and evidence stay immutable.
STRATEGY_RESEARCH_V23_TARGET_RUNTIME_BUNDLE_SHA256 = (
    "18c2964083d39c7abc761637f173b94eb72c6c5fff4be762177ca39048792558"
)
STRATEGY_RESEARCH_V24_TARGET_RECIPE_VERSION = (
    "qlib-rdagent-single-mainline-2026-09-03-v24"
)
STRATEGY_RESEARCH_V24_TARGET_RUNNER_SHA256 = (
    "79ff5cee1f046dea95bba13358005d975f0a83d7324c8f0fe18b07e5962b9615"
)
# Historical v24 seal. It remains independently addressable after v25 becomes
# current so already-applied database constraints and evidence stay immutable.
STRATEGY_RESEARCH_V24_TARGET_RUNTIME_BUNDLE_SHA256 = (
    "82b151261f4fff98a7848f093fb0ea69e47fadf10b41df83288279e7a076287e"
)
STRATEGY_RESEARCH_V25_TARGET_RECIPE_VERSION = (
    "qlib-rdagent-single-mainline-2026-09-03-v25"
)
STRATEGY_RESEARCH_V25_TARGET_RUNNER_SHA256 = (
    "79ff5cee1f046dea95bba13358005d975f0a83d7324c8f0fe18b07e5962b9615"
)
# Historical v25 seal. It remains independently addressable after v26 becomes
# current so already-applied database constraints and evidence stay immutable.
STRATEGY_RESEARCH_V25_TARGET_RUNTIME_BUNDLE_SHA256 = (
    "e9f26125b1675695b708ee48b8bacf65bea8c83abb56ac7741d799a407f48db1"
)
STRATEGY_RESEARCH_V26_TARGET_RECIPE_VERSION = (
    "qlib-rdagent-single-mainline-2026-09-03-v26"
)
STRATEGY_RESEARCH_V26_TARGET_RUNNER_SHA256 = (
    "79ff5cee1f046dea95bba13358005d975f0a83d7324c8f0fe18b07e5962b9615"
)
# Historical v26 seal. It remains independently addressable after v27 becomes
# current so already-applied database constraints and evidence stay immutable.
STRATEGY_RESEARCH_V26_TARGET_RUNTIME_BUNDLE_SHA256 = (
    "6a584023edd2b8589ac0bb2dcadf5dc7e06f6999b606aded11459ebc401aa9b4"
)
STRATEGY_RESEARCH_V27_TARGET_RECIPE_VERSION = (
    "qlib-rdagent-single-mainline-2026-09-04-v27"
)
STRATEGY_RESEARCH_V27_TARGET_RUNNER_SHA256 = (
    "79ff5cee1f046dea95bba13358005d975f0a83d7324c8f0fe18b07e5962b9615"
)
# Historical v27 seal. It remains independently addressable after v28 becomes
# current so already-applied database constraints and evidence stay immutable.
STRATEGY_RESEARCH_V27_TARGET_RUNTIME_BUNDLE_SHA256 = (
    "c987c4fb6f1f5ac6f4d8cf21f3f901ef1f8c4b8ee33bb7153b8afa3c46162697"
)
STRATEGY_RESEARCH_V28_TARGET_RECIPE_VERSION = (
    "qlib-rdagent-single-mainline-2026-09-04-v28"
)
STRATEGY_RESEARCH_V28_TARGET_RUNNER_SHA256 = (
    "79ff5cee1f046dea95bba13358005d975f0a83d7324c8f0fe18b07e5962b9615"
)
# Historical v28 seal. It remains independently addressable after v29 becomes
# current so already-applied database constraints and evidence stay immutable.
STRATEGY_RESEARCH_V28_TARGET_RUNTIME_BUNDLE_SHA256 = (
    "d8bb1e2ba21413e4cfb04ab819a270e4699f7d5e9964939ae727aa8055876b85"
)
STRATEGY_RESEARCH_V29_TARGET_RECIPE_VERSION = (
    "qlib-rdagent-single-mainline-2026-09-04-v29"
)
STRATEGY_RESEARCH_V29_TARGET_RUNNER_SHA256 = (
    "79ff5cee1f046dea95bba13358005d975f0a83d7324c8f0fe18b07e5962b9615"
)
# Historical v29 seal. It remains independently addressable after v30 becomes
# current so already-applied database constraints and evidence stay immutable.
STRATEGY_RESEARCH_V29_TARGET_RUNTIME_BUNDLE_SHA256 = (
    "2861feb4ffe72969cda17b77fe2cc8928b951b37e6247836043e5ef3a62eb705"
)
STRATEGY_RESEARCH_V30_TARGET_RECIPE_VERSION = (
    "qlib-rdagent-single-mainline-2026-09-04-v30"
)
STRATEGY_RESEARCH_V30_TARGET_RUNNER_SHA256 = (
    "f1ff1ab72165465675d8e699215fd0082c8dedbab0e12d910dc45485a470a457"
)
# Historical v30 seal. It remains independently addressable after v31 becomes
# current so already-applied database constraints and evidence stay immutable.
STRATEGY_RESEARCH_V30_TARGET_RUNTIME_BUNDLE_SHA256 = (
    "25d544bf2527cf4ca8d890b3e3a15836c87f5e6f28c75573ed7027da403171a1"
)
STRATEGY_RESEARCH_TARGET_RECIPE_VERSION = (
    "qlib-rdagent-single-mainline-2026-09-04-v31"
)
STRATEGY_RESEARCH_TARGET_RUNNER_SHA256 = (
    "036d7979581b3de719532303435f8fba81e43f4403b3c82bf371e5c18820b7f2"
)
# Filled after the complete v31 source closure is stable. The source-closure
# normalizer excludes both v31 seal assignments from their own digest.
STRATEGY_RESEARCH_TARGET_RUNTIME_BUNDLE_SHA256 = (
    "dac80be45fcae33d2bcdafff76aa2b71feac6624deb25b542bfc02961584f1b8"
)
TRANSPARENT_BASELINE_RUNNER_FIELD = "target_runner_sha256"
TRANSPARENT_BASELINE_JOB_RUNNER_FIELD = "transparent_baseline_runner_sha256"
TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD = "target_runtime_bundle_sha256"
TRANSPARENT_BASELINE_JOB_RUNTIME_BUNDLE_FIELD = (
    "transparent_baseline_runtime_bundle_sha256"
)
TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD = (
    "target_worker_runtime_image_digest"
)
TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD = (
    "transparent_baseline_worker_runtime_image_digest"
)
TRANSPARENT_BASELINE_RESULT_WORKER_RUNTIME_IMAGE_FIELD = (
    "worker_runtime_image_digest"
)
WORKER_RUNTIME_IMAGE_DIGEST_ENV = "QUANTLAB_WORKER_RUNTIME_IMAGE_DIGEST"
_IMAGE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_TRANSPARENT_RECIPE_IDS = {
    "short_relative_strength",
    "swing_trend",
    "long_quality_value",
}
_TARGET_RUNNERS = {
    OPTIMIZER_APPLICABILITY_TARGET_RECIPE_VERSION: (
        OPTIMIZER_APPLICABILITY_TARGET_RUNNER_SHA256
    ),
    CANONICAL_LF_TARGET_RECIPE_VERSION: CANONICAL_LF_TARGET_RUNNER_SHA256,
    RUNTIME_ALIGNMENT_TARGET_RECIPE_VERSION: RUNTIME_ALIGNMENT_TARGET_RUNNER_SHA256,
    RUNTIME_INPUT_SCOPE_TARGET_RECIPE_VERSION: RUNTIME_INPUT_SCOPE_TARGET_RUNNER_SHA256,
    POSITION_RISK_TARGET_RECIPE_VERSION: POSITION_RISK_TARGET_RUNNER_SHA256,
    FAIL_CLOSED_EXECUTION_TARGET_RECIPE_VERSION: (
        FAIL_CLOSED_EXECUTION_TARGET_RUNNER_SHA256
    ),
    FILL_AWARE_HOLDING_AGE_TARGET_RECIPE_VERSION: (
        FILL_AWARE_HOLDING_AGE_TARGET_RUNNER_SHA256
    ),
    SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RECIPE_VERSION: (
        SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RUNNER_SHA256
    ),
    DISCRETE_MAX_POSITION_REPAIR_TARGET_RECIPE_VERSION: (
        DISCRETE_MAX_POSITION_REPAIR_TARGET_RUNNER_SHA256
    ),
    TOPK_INDUSTRY_CAPACITY_REPAIR_TARGET_RECIPE_VERSION: (
        TOPK_INDUSTRY_CAPACITY_REPAIR_TARGET_RUNNER_SHA256
    ),
    FORWARD_ONLY_REHABILITATION_TARGET_RECIPE_VERSION: (
        FORWARD_ONLY_REHABILITATION_TARGET_RUNNER_SHA256
    ),
    STRATEGY_RESEARCH_V20_TARGET_RECIPE_VERSION: (
        STRATEGY_RESEARCH_V20_TARGET_RUNNER_SHA256
    ),
    STRATEGY_RESEARCH_V21_TARGET_RECIPE_VERSION: (
        STRATEGY_RESEARCH_V21_TARGET_RUNNER_SHA256
    ),
    STRATEGY_RESEARCH_V22_TARGET_RECIPE_VERSION: (
        STRATEGY_RESEARCH_V22_TARGET_RUNNER_SHA256
    ),
    STRATEGY_RESEARCH_V23_TARGET_RECIPE_VERSION: (
        STRATEGY_RESEARCH_V23_TARGET_RUNNER_SHA256
    ),
    STRATEGY_RESEARCH_V24_TARGET_RECIPE_VERSION: (
        STRATEGY_RESEARCH_V24_TARGET_RUNNER_SHA256
    ),
    STRATEGY_RESEARCH_V25_TARGET_RECIPE_VERSION: (
        STRATEGY_RESEARCH_V25_TARGET_RUNNER_SHA256
    ),
    STRATEGY_RESEARCH_V26_TARGET_RECIPE_VERSION: (
        STRATEGY_RESEARCH_V26_TARGET_RUNNER_SHA256
    ),
    STRATEGY_RESEARCH_V27_TARGET_RECIPE_VERSION: (
        STRATEGY_RESEARCH_V27_TARGET_RUNNER_SHA256
    ),
    STRATEGY_RESEARCH_V28_TARGET_RECIPE_VERSION: (
        STRATEGY_RESEARCH_V28_TARGET_RUNNER_SHA256
    ),
    STRATEGY_RESEARCH_V29_TARGET_RECIPE_VERSION: (
        STRATEGY_RESEARCH_V29_TARGET_RUNNER_SHA256
    ),
    STRATEGY_RESEARCH_V30_TARGET_RECIPE_VERSION: (
        STRATEGY_RESEARCH_V30_TARGET_RUNNER_SHA256
    ),
    STRATEGY_RESEARCH_TARGET_RECIPE_VERSION: STRATEGY_RESEARCH_TARGET_RUNNER_SHA256,
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_bundle_sha256(project_root: Path, paths: tuple[str, ...]) -> str:
    inventory = []
    for relative in paths:
        path = project_root / relative
        payload = path.read_bytes()
        inventory.append(
            {
                "path": relative,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    encoded = json.dumps(
        inventory,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def runtime_alignment_bundle_sha256(project_root: Path) -> str:
    """Hash the immutable five-module v10/v11 runtime inventory."""

    return _runtime_bundle_sha256(project_root, RUNTIME_ALIGNMENT_RUNTIME_PATHS)


def position_risk_bundle_sha256(project_root: Path) -> str:
    """Hash the fail-closed local Python closure for the current material runtime."""

    return position_risk_source_closure_sha256(project_root)


def target_runner_for_recipe(recipe_id: Any, recipe_version: Any) -> str | None:
    normalized_recipe_id = str(recipe_id or "")
    normalized_recipe_version = str(recipe_version or "")
    if normalized_recipe_id not in _TRANSPARENT_RECIPE_IDS:
        return None
    if (
        normalized_recipe_version
        == FORWARD_ONLY_REHABILITATION_TARGET_RECIPE_VERSION
        and normalized_recipe_id != "short_relative_strength"
    ):
        return None
    return _TARGET_RUNNERS.get(normalized_recipe_version)


def target_runtime_bundle_for_recipe(recipe_id: Any, recipe_version: Any) -> str | None:
    normalized_recipe_id = str(recipe_id or "")
    normalized_recipe_version = str(recipe_version or "")
    if normalized_recipe_id not in _TRANSPARENT_RECIPE_IDS:
        return None
    if (
        normalized_recipe_version
        == FORWARD_ONLY_REHABILITATION_TARGET_RECIPE_VERSION
        and normalized_recipe_id != "short_relative_strength"
    ):
        return None
    if normalized_recipe_version == RUNTIME_ALIGNMENT_TARGET_RECIPE_VERSION:
        return RUNTIME_ALIGNMENT_TARGET_RUNTIME_BUNDLE_SHA256
    if str(recipe_version or "") == RUNTIME_INPUT_SCOPE_TARGET_RECIPE_VERSION:
        return RUNTIME_INPUT_SCOPE_TARGET_RUNTIME_BUNDLE_SHA256
    if str(recipe_version or "") == POSITION_RISK_TARGET_RECIPE_VERSION:
        return POSITION_RISK_TARGET_RUNTIME_BUNDLE_SHA256
    if str(recipe_version or "") == FAIL_CLOSED_EXECUTION_TARGET_RECIPE_VERSION:
        return FAIL_CLOSED_EXECUTION_TARGET_RUNTIME_BUNDLE_SHA256
    if str(recipe_version or "") == FILL_AWARE_HOLDING_AGE_TARGET_RECIPE_VERSION:
        return FILL_AWARE_HOLDING_AGE_TARGET_RUNTIME_BUNDLE_SHA256
    if (
        str(recipe_version or "")
        == SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RECIPE_VERSION
    ):
        return SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RUNTIME_BUNDLE_SHA256
    if str(recipe_version or "") == DISCRETE_MAX_POSITION_REPAIR_TARGET_RECIPE_VERSION:
        return DISCRETE_MAX_POSITION_REPAIR_TARGET_RUNTIME_BUNDLE_SHA256
    if (
        str(recipe_version or "")
        == TOPK_INDUSTRY_CAPACITY_REPAIR_TARGET_RECIPE_VERSION
    ):
        return TOPK_INDUSTRY_CAPACITY_REPAIR_TARGET_RUNTIME_BUNDLE_SHA256
    if normalized_recipe_version == FORWARD_ONLY_REHABILITATION_TARGET_RECIPE_VERSION:
        return FORWARD_ONLY_REHABILITATION_TARGET_RUNTIME_BUNDLE_SHA256
    if normalized_recipe_version == STRATEGY_RESEARCH_V20_TARGET_RECIPE_VERSION:
        return STRATEGY_RESEARCH_V20_TARGET_RUNTIME_BUNDLE_SHA256
    if normalized_recipe_version == STRATEGY_RESEARCH_V21_TARGET_RECIPE_VERSION:
        return STRATEGY_RESEARCH_V21_TARGET_RUNTIME_BUNDLE_SHA256
    if normalized_recipe_version == STRATEGY_RESEARCH_V22_TARGET_RECIPE_VERSION:
        return STRATEGY_RESEARCH_V22_TARGET_RUNTIME_BUNDLE_SHA256
    if normalized_recipe_version == STRATEGY_RESEARCH_V23_TARGET_RECIPE_VERSION:
        return STRATEGY_RESEARCH_V23_TARGET_RUNTIME_BUNDLE_SHA256
    if normalized_recipe_version == STRATEGY_RESEARCH_V24_TARGET_RECIPE_VERSION:
        return STRATEGY_RESEARCH_V24_TARGET_RUNTIME_BUNDLE_SHA256
    if normalized_recipe_version == STRATEGY_RESEARCH_V25_TARGET_RECIPE_VERSION:
        return STRATEGY_RESEARCH_V25_TARGET_RUNTIME_BUNDLE_SHA256
    if normalized_recipe_version == STRATEGY_RESEARCH_V26_TARGET_RECIPE_VERSION:
        return STRATEGY_RESEARCH_V26_TARGET_RUNTIME_BUNDLE_SHA256
    if normalized_recipe_version == STRATEGY_RESEARCH_V27_TARGET_RECIPE_VERSION:
        return STRATEGY_RESEARCH_V27_TARGET_RUNTIME_BUNDLE_SHA256
    if normalized_recipe_version == STRATEGY_RESEARCH_V28_TARGET_RECIPE_VERSION:
        return STRATEGY_RESEARCH_V28_TARGET_RUNTIME_BUNDLE_SHA256
    if normalized_recipe_version == STRATEGY_RESEARCH_V29_TARGET_RECIPE_VERSION:
        return STRATEGY_RESEARCH_V29_TARGET_RUNTIME_BUNDLE_SHA256
    if normalized_recipe_version == STRATEGY_RESEARCH_V30_TARGET_RECIPE_VERSION:
        return STRATEGY_RESEARCH_V30_TARGET_RUNTIME_BUNDLE_SHA256
    if normalized_recipe_version == STRATEGY_RESEARCH_TARGET_RECIPE_VERSION:
        return STRATEGY_RESEARCH_TARGET_RUNTIME_BUNDLE_SHA256
    return None


def target_worker_runtime_image_for_recipe(
    recipe_id: Any,
    recipe_version: Any,
) -> str | None:
    """Return the exact release-built worker image required by material v12+.

    The digest is deliberately release data rather than a source constant: the
    release controller resolves Docker's immutable content ID after building
    the worker and stamps it into every process before any sealed version exists.
    """

    normalized_recipe_id = str(recipe_id or "")
    normalized_recipe_version = str(recipe_version or "")
    if (
        normalized_recipe_id not in _TRANSPARENT_RECIPE_IDS
        or (
            normalized_recipe_version
            == FORWARD_ONLY_REHABILITATION_TARGET_RECIPE_VERSION
            and normalized_recipe_id != "short_relative_strength"
        )
        or normalized_recipe_version
        not in {
            POSITION_RISK_TARGET_RECIPE_VERSION,
            FAIL_CLOSED_EXECUTION_TARGET_RECIPE_VERSION,
            FILL_AWARE_HOLDING_AGE_TARGET_RECIPE_VERSION,
            SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RECIPE_VERSION,
            DISCRETE_MAX_POSITION_REPAIR_TARGET_RECIPE_VERSION,
            TOPK_INDUSTRY_CAPACITY_REPAIR_TARGET_RECIPE_VERSION,
            FORWARD_ONLY_REHABILITATION_TARGET_RECIPE_VERSION,
            STRATEGY_RESEARCH_V20_TARGET_RECIPE_VERSION,
            STRATEGY_RESEARCH_V21_TARGET_RECIPE_VERSION,
            STRATEGY_RESEARCH_V22_TARGET_RECIPE_VERSION,
            STRATEGY_RESEARCH_V23_TARGET_RECIPE_VERSION,
            STRATEGY_RESEARCH_V24_TARGET_RECIPE_VERSION,
            STRATEGY_RESEARCH_V25_TARGET_RECIPE_VERSION,
            STRATEGY_RESEARCH_V26_TARGET_RECIPE_VERSION,
            STRATEGY_RESEARCH_V27_TARGET_RECIPE_VERSION,
            STRATEGY_RESEARCH_V28_TARGET_RECIPE_VERSION,
            STRATEGY_RESEARCH_V29_TARGET_RECIPE_VERSION,
            STRATEGY_RESEARCH_V30_TARGET_RECIPE_VERSION,
            STRATEGY_RESEARCH_TARGET_RECIPE_VERSION,
        }
    ):
        return None
    version_label = {
        POSITION_RISK_TARGET_RECIPE_VERSION: "v12",
        FAIL_CLOSED_EXECUTION_TARGET_RECIPE_VERSION: "v13",
        FILL_AWARE_HOLDING_AGE_TARGET_RECIPE_VERSION: "v14",
        SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RECIPE_VERSION: "v15",
        DISCRETE_MAX_POSITION_REPAIR_TARGET_RECIPE_VERSION: "v16",
        TOPK_INDUSTRY_CAPACITY_REPAIR_TARGET_RECIPE_VERSION: "v17",
        FORWARD_ONLY_REHABILITATION_TARGET_RECIPE_VERSION: "v18",
        STRATEGY_RESEARCH_V20_TARGET_RECIPE_VERSION: "v20",
        STRATEGY_RESEARCH_V21_TARGET_RECIPE_VERSION: "v21",
        STRATEGY_RESEARCH_V22_TARGET_RECIPE_VERSION: "v22",
        STRATEGY_RESEARCH_V23_TARGET_RECIPE_VERSION: "v23",
        STRATEGY_RESEARCH_V24_TARGET_RECIPE_VERSION: "v24",
        STRATEGY_RESEARCH_V25_TARGET_RECIPE_VERSION: "v25",
        STRATEGY_RESEARCH_V26_TARGET_RECIPE_VERSION: "v26",
        STRATEGY_RESEARCH_V27_TARGET_RECIPE_VERSION: "v27",
        STRATEGY_RESEARCH_V28_TARGET_RECIPE_VERSION: "v28",
        STRATEGY_RESEARCH_V29_TARGET_RECIPE_VERSION: "v29",
        STRATEGY_RESEARCH_V30_TARGET_RECIPE_VERSION: "v30",
        STRATEGY_RESEARCH_TARGET_RECIPE_VERSION: "v31",
    }[normalized_recipe_version]
    value = str(os.getenv(WORKER_RUNTIME_IMAGE_DIGEST_ENV) or "").strip().lower()
    if not _IMAGE_DIGEST.fullmatch(value):
        raise ValueError(
            f"transparent {version_label} worker runtime image digest is missing or invalid"
        )
    return value


def bind_transparent_baseline_job_identity(
    *,
    config: Mapping[str, Any],
    job_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze the governed runtime identity into a child evaluation job.

    Transparent strategies are executable only when the immutable strategy
    bootstrap, the queued job, and the worker release all name the same
    runner, source closure, and worker image.  Constructing these fields in one
    place prevents nested parameter experiments from silently dropping the
    identity that ordinary formal backtests already carry.
    """

    payload = dict(job_payload)
    recipe_id = config.get("recipe_id")
    recipe_version = config.get("recipe_version")
    expected_runner = target_runner_for_recipe(recipe_id, recipe_version)
    expected_bundle = target_runtime_bundle_for_recipe(recipe_id, recipe_version)
    bootstrap_raw = config.get("transparent_baseline_bootstrap")
    bootstrap = dict(bootstrap_raw) if isinstance(bootstrap_raw, Mapping) else {}
    governed_fields = (
        (
            TRANSPARENT_BASELINE_RUNNER_FIELD,
            TRANSPARENT_BASELINE_JOB_RUNNER_FIELD,
            expected_runner,
        ),
        (
            TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD,
            TRANSPARENT_BASELINE_JOB_RUNTIME_BUNDLE_FIELD,
            expected_bundle,
        ),
    )
    if expected_runner is None:
        if any(
            bootstrap.get(source_field) is not None
            or payload.get(job_field) is not None
            for source_field, job_field, _ in governed_fields
        ) or (
            bootstrap.get(TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD) is not None
            or payload.get(TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD)
            is not None
        ):
            raise ValueError(
                "runner repair identity is forbidden outside governed transparent recipes"
            )
        return payload

    expected_worker_image = target_worker_runtime_image_for_recipe(
        recipe_id, recipe_version
    )
    for source_field, job_field, expected in governed_fields:
        value = bootstrap.get(source_field)
        if value != expected:
            raise ValueError("transparent strategy bootstrap runtime identity is invalid")
        submitted = payload.get(job_field)
        if submitted is not None and submitted != value:
            raise ValueError("transparent strategy job runtime identity changed")
        payload[job_field] = value
    bootstrap_worker_image = bootstrap.get(
        TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD
    )
    if bootstrap_worker_image != expected_worker_image:
        raise ValueError("transparent strategy bootstrap worker image is invalid")
    submitted_worker_image = payload.get(
        TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD
    )
    if (
        submitted_worker_image is not None
        and submitted_worker_image != expected_worker_image
    ):
        raise ValueError("transparent strategy job worker image changed")
    payload[TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD] = expected_worker_image
    return payload


def require_transparent_baseline_runner(
    *,
    config: Mapping[str, Any],
    job_payload: Mapping[str, Any],
    runner_path: Path,
) -> str | None:
    """Verify bootstrap, job and executed runner have one governed byte identity."""

    bootstrap_raw = config.get("transparent_baseline_bootstrap")
    bootstrap = dict(bootstrap_raw) if isinstance(bootstrap_raw, Mapping) else {}
    expected = target_runner_for_recipe(
        config.get("recipe_id"), config.get("recipe_version")
    )
    expected_bundle = target_runtime_bundle_for_recipe(
        config.get("recipe_id"), config.get("recipe_version")
    )
    bootstrap_value = bootstrap.get(TRANSPARENT_BASELINE_RUNNER_FIELD)
    payload_value = job_payload.get(TRANSPARENT_BASELINE_JOB_RUNNER_FIELD)
    bootstrap_bundle = bootstrap.get(TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD)
    payload_bundle = job_payload.get(TRANSPARENT_BASELINE_JOB_RUNTIME_BUNDLE_FIELD)
    bootstrap_worker_image = bootstrap.get(
        TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD
    )
    payload_worker_image = job_payload.get(
        TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD
    )
    if expected is None:
        if any(
            value is not None
            for value in (
                bootstrap_value,
                payload_value,
                bootstrap_bundle,
                payload_bundle,
                bootstrap_worker_image,
                payload_worker_image,
            )
        ):
            raise ValueError(
                "runner repair identity is forbidden outside governed transparent recipes"
            )
        return None
    version_label = {
        OPTIMIZER_APPLICABILITY_TARGET_RECIPE_VERSION: "v8",
        CANONICAL_LF_TARGET_RECIPE_VERSION: "v9",
        RUNTIME_ALIGNMENT_TARGET_RECIPE_VERSION: "v10",
        RUNTIME_INPUT_SCOPE_TARGET_RECIPE_VERSION: "v11",
        POSITION_RISK_TARGET_RECIPE_VERSION: "v12",
        FAIL_CLOSED_EXECUTION_TARGET_RECIPE_VERSION: "v13",
        FILL_AWARE_HOLDING_AGE_TARGET_RECIPE_VERSION: "v14",
        SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RECIPE_VERSION: "v15",
        DISCRETE_MAX_POSITION_REPAIR_TARGET_RECIPE_VERSION: "v16",
        TOPK_INDUSTRY_CAPACITY_REPAIR_TARGET_RECIPE_VERSION: "v17",
        FORWARD_ONLY_REHABILITATION_TARGET_RECIPE_VERSION: "v18",
        STRATEGY_RESEARCH_V20_TARGET_RECIPE_VERSION: "v20",
        STRATEGY_RESEARCH_V21_TARGET_RECIPE_VERSION: "v21",
        STRATEGY_RESEARCH_V22_TARGET_RECIPE_VERSION: "v22",
        STRATEGY_RESEARCH_V23_TARGET_RECIPE_VERSION: "v23",
        STRATEGY_RESEARCH_V24_TARGET_RECIPE_VERSION: "v24",
        STRATEGY_RESEARCH_V25_TARGET_RECIPE_VERSION: "v25",
        STRATEGY_RESEARCH_V26_TARGET_RECIPE_VERSION: "v26",
        STRATEGY_RESEARCH_V27_TARGET_RECIPE_VERSION: "v27",
        STRATEGY_RESEARCH_V28_TARGET_RECIPE_VERSION: "v28",
        STRATEGY_RESEARCH_V29_TARGET_RECIPE_VERSION: "v29",
        STRATEGY_RESEARCH_V30_TARGET_RECIPE_VERSION: "v30",
        STRATEGY_RESEARCH_TARGET_RECIPE_VERSION: "v31",
    }[str(config.get("recipe_version") or "")]
    if bootstrap_value != expected or payload_value != expected:
        raise ValueError(
            f"transparent {version_label} runner identity differs from its sealed bootstrap"
        )
    if (
        bootstrap_bundle != expected_bundle
        or payload_bundle != expected_bundle
    ):
        raise ValueError(
            f"transparent {version_label} runtime bundle identity differs from its "
            "sealed bootstrap"
        )
    expected_worker_image = target_worker_runtime_image_for_recipe(
        config.get("recipe_id"), config.get("recipe_version")
    )
    if expected_worker_image is not None and (
        bootstrap_worker_image != expected_worker_image
        or payload_worker_image != expected_worker_image
    ):
        raise ValueError(
            f"transparent {version_label} worker runtime image differs from its "
            "sealed bootstrap"
        )
    try:
        observed = _file_sha256(runner_path)
    except OSError as exc:
        raise ValueError(
            f"transparent {version_label} runner cannot be verified"
        ) from exc
    if observed != expected:
        raise ValueError(
            f"transparent {version_label} runner bytes differ from the repair authorization"
        )
    if version_label in {
        "v10",
        "v11",
        "v12",
        "v13",
        "v14",
        "v15",
        "v16",
        "v17",
        "v18",
        "v20",
        "v21",
        "v22",
        "v23",
        "v24",
        "v25",
        "v26",
        "v27",
        "v28",
    }:
        try:
            bundle_sha256 = (
                position_risk_bundle_sha256(runner_path.parents[1])
                if version_label
                in {
                    "v12",
                    "v13",
                    "v14",
                    "v15",
                    "v16",
                    "v17",
                    "v18",
                    "v20",
                    "v21",
                    "v22",
                    "v23",
                    "v24",
                    "v25",
                    "v26",
                    "v27",
                    "v28",
                }
                else runtime_alignment_bundle_sha256(runner_path.parents[1])
            )
        except OSError as exc:
            raise ValueError(
                f"transparent {version_label} runtime bundle cannot be verified"
            ) from exc
        expected_bundle_sha256 = {
            "v10": RUNTIME_ALIGNMENT_TARGET_RUNTIME_BUNDLE_SHA256,
            "v11": RUNTIME_INPUT_SCOPE_TARGET_RUNTIME_BUNDLE_SHA256,
            "v12": POSITION_RISK_TARGET_RUNTIME_BUNDLE_SHA256,
            "v13": FAIL_CLOSED_EXECUTION_TARGET_RUNTIME_BUNDLE_SHA256,
            "v14": FILL_AWARE_HOLDING_AGE_TARGET_RUNTIME_BUNDLE_SHA256,
            "v15": SINGLE_MEMBER_PRE_RESULT_REPAIR_TARGET_RUNTIME_BUNDLE_SHA256,
            "v16": DISCRETE_MAX_POSITION_REPAIR_TARGET_RUNTIME_BUNDLE_SHA256,
            "v17": TOPK_INDUSTRY_CAPACITY_REPAIR_TARGET_RUNTIME_BUNDLE_SHA256,
            "v18": FORWARD_ONLY_REHABILITATION_TARGET_RUNTIME_BUNDLE_SHA256,
            "v20": STRATEGY_RESEARCH_V20_TARGET_RUNTIME_BUNDLE_SHA256,
            "v21": STRATEGY_RESEARCH_V21_TARGET_RUNTIME_BUNDLE_SHA256,
            "v22": STRATEGY_RESEARCH_V22_TARGET_RUNTIME_BUNDLE_SHA256,
            "v23": STRATEGY_RESEARCH_V23_TARGET_RUNTIME_BUNDLE_SHA256,
            "v24": STRATEGY_RESEARCH_V24_TARGET_RUNTIME_BUNDLE_SHA256,
            "v25": STRATEGY_RESEARCH_V25_TARGET_RUNTIME_BUNDLE_SHA256,
            "v26": STRATEGY_RESEARCH_V26_TARGET_RUNTIME_BUNDLE_SHA256,
            "v27": STRATEGY_RESEARCH_V27_TARGET_RUNTIME_BUNDLE_SHA256,
            "v28": STRATEGY_RESEARCH_V28_TARGET_RUNTIME_BUNDLE_SHA256,
            "v29": STRATEGY_RESEARCH_V29_TARGET_RUNTIME_BUNDLE_SHA256,
            "v30": STRATEGY_RESEARCH_V30_TARGET_RUNTIME_BUNDLE_SHA256,
            "v31": STRATEGY_RESEARCH_TARGET_RUNTIME_BUNDLE_SHA256,
        }[version_label]
        if bundle_sha256 != expected_bundle_sha256:
            raise ValueError(
                f"transparent {version_label} runtime bundle differs from the repair "
                "authorization"
            )
    return observed
