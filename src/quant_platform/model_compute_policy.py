"""Frozen compute envelopes shared by scheduling, preparation and model execution.

The full table identifies the experiment environment.  Its selected row is
execution evidence; model seeds and completed cells never alter this table.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

MODEL_COMPUTE_POLICY_VERSION = "model-compute-policy-v1-profile-grid"
MODEL_RESOURCE_POLICY_VERSION = "model-resource-policy-v9-progress-observed"
DEFAULT_COMPUTE_PROFILE = "cpu4"


def fixed_model_cell_grid_policy() -> dict[str, Any]:
    return {
        "contract_version": MODEL_COMPUTE_POLICY_VERSION,
        "batch_limits": {"cpu_count": 8, "memory_gb": 40, "max_cells": 2},
        "compute_profile": DEFAULT_COMPUTE_PROFILE,
        "engine_compute_profiles": {"lightgbm_baseline": "cpu8"},
        "compute_profiles": {
            f"cpu{count}": {
                "cpu_count": count, "compute_threads": count, "blas_threads": count,
                "torch_interop_threads": 1, "dataloader_workers": 0,
            }
            for count in (2, 4, 8)
        },
        "memory_profiles_gb": {
            "ridge_baseline": {"recent_3y": 40, "balanced_5y": 32, "robust_10y": 16},
        },
        "fallback_memory_gb": 40,
        "preparation_threads": 1,
        "qlib_kernels": 1,
        "swap_enabled": False,
        "exclusive_engines": ["platform_transformer"],
    }


def governed_cell_resource_allocation(manifest: Mapping[str, Any]) -> dict[str, Any]:
    policy = fixed_model_cell_grid_policy()
    supplied_policy = manifest.get("model_cell_grid_policy")
    if supplied_policy is not None and supplied_policy != policy:
        raise ValueError("model cell grid policy differs from the frozen runtime policy")
    engine = str(manifest.get("model_engine") or "rdagent_pytorch")
    profile = str(manifest.get("evaluation_profile_id") or "unprofiled")
    stage = str(manifest.get("resource_stage") or "full_validation")
    compute_profile = policy["engine_compute_profiles"].get(engine, policy["compute_profile"])
    memory = policy["fallback_memory_gb"]
    if stage in {"screening", "full_validation"}:
        memory = policy["memory_profiles_gb"].get(engine, {}).get(profile, memory)
    allocation = {
        "contract_version": MODEL_COMPUTE_POLICY_VERSION,
        "evaluation_profile_id": profile,
        "model_engine": engine,
        "resource_stage": stage,
        "compute_profile": compute_profile,
        **policy["compute_profiles"][compute_profile],
        "memory_gb": memory,
        "exclusive": engine in policy["exclusive_engines"],
    }
    supplied = manifest.get("model_cell_allocation")
    if supplied is not None and supplied != allocation:
        raise ValueError("model cell allocation differs from its governed profile")
    return allocation


def model_thread_environment(allocation: Mapping[str, Any], *, preparation=False) -> dict[str, str]:
    # All native libraries see an explicit limit before their first import.
    threads = 1 if preparation else int(allocation["blas_threads"])
    return {
        "OMP_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "VECLIB_MAXIMUM_THREADS": str(threads),
        "NUMEXPR_NUM_THREADS": str(threads),
        "BLIS_NUM_THREADS": str(threads),
        "OMP_DYNAMIC": "FALSE",
        "MKL_DYNAMIC": "FALSE",
    }


def require_model_thread_environment(allocation: Mapping[str, Any], environ: Mapping[str, str]):
    expected = model_thread_environment(allocation)
    if any(environ.get(name) != value for name, value in expected.items()):
        raise ValueError("model native thread environment differs from its governed allocation")
    return expected
