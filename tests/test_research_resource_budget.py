from __future__ import annotations

from pathlib import Path

import pytest

from quant_platform.job_store import (
    MAX_NUMERICAL_THREADS_PER_JOB,
    research_job_cpu_cost,
    research_job_memory_gb,
)

pytestmark = pytest.mark.no_database


def test_research_resource_costs_match_the_governed_queue_contract() -> None:
    assert MAX_NUMERICAL_THREADS_PER_JOB == 8
    assert research_job_cpu_cost("data_qlib") == 16
    assert research_job_memory_gb("data_qlib") == 24
    assert research_job_cpu_cost("minute_qlib") == 16
    assert research_job_memory_gb("minute_qlib") == 24
    assert research_job_cpu_cost("qlib_baseline") == 8
    assert research_job_memory_gb("qlib_baseline") == 40
    assert research_job_cpu_cost("rdagent_factor") == 8
    assert research_job_memory_gb("rdagent_factor") == 12
    assert research_job_cpu_cost("rdagent_model") == 8
    assert research_job_memory_gb("rdagent_model") == 16
    assert research_job_cpu_cost("rdagent_factor_report") == 4
    assert research_job_memory_gb("rdagent_factor_report") == 8
    assert research_job_cpu_cost("rdagent_quant") == 12
    assert research_job_memory_gb("rdagent_quant") == 20
    assert research_job_memory_gb("factor_sota_evaluate") == 40
    assert research_job_memory_gb("model_evaluate") == 40
    assert research_job_memory_gb("quant_bundle_evaluate") == 40
    assert research_job_memory_gb("strategy_backtest") == 40
    assert research_job_cpu_cost("strategy_health_collect") == 4
    assert research_job_memory_gb("strategy_health_collect") == 8
    assert (
        research_job_memory_gb("qlib_baseline")
        + research_job_memory_gb("model_evaluate")
        > 40
    )
    assert (
        research_job_memory_gb("qlib_baseline")
        + research_job_memory_gb("external_factor_evaluate")
        > 40
    )
    assert research_job_cpu_cost("model_refit") == 0
    assert research_job_memory_gb("simulation_replay") == 0


def test_all_cpu_research_workers_share_one_host_budget() -> None:
    compose = (Path(__file__).parents[1] / "deploy" / "compose.yaml").read_text(
        encoding="utf-8"
    )

    assert compose.count('RESEARCH_CPU_BUDGET: "24"') == 6
    assert compose.count('RESEARCH_MEMORY_BUDGET_GB: "40"') == 6
    assert 'WORKER_JOB_KINDS: model_refit,recommendation_refresh,' in compose


def test_primary_data_worker_has_hard_limits_and_numerical_thread_caps() -> None:
    compose = (Path(__file__).parents[1] / "deploy" / "compose.yaml").read_text(
        encoding="utf-8"
    )
    worker_block = compose.split("\n  worker:\n", 1)[1].split(
        "\n  evaluation-worker:\n", 1
    )[0]

    assert 'cpus: "16.0"' in worker_block
    assert "mem_limit: 32g" in worker_block
    assert "pids_limit: 512" in worker_block
    assert 'WORKER_CONCURRENCY: "1"' in worker_block
    assert 'RESEARCH_CPU_BUDGET: "24"' in worker_block
    assert 'RESEARCH_MEMORY_BUDGET_GB: "40"' in worker_block
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMEXPR_MAX_THREADS",
    ):
        assert f'{variable}: "8"' in worker_block


def test_evaluation_worker_keeps_host_capacity_but_not_a_48gb_ledger() -> None:
    compose = (Path(__file__).parents[1] / "deploy" / "compose.yaml").read_text(
        encoding="utf-8"
    )
    evaluation_block = compose.split("\n  evaluation-worker:\n", 1)[1].split(
        "\n  paper-worker:\n", 1
    )[0]

    assert 'cpus: "24.0"' in evaluation_block
    assert "mem_limit: 48g" in evaluation_block
    assert 'WORKER_CONCURRENCY: "3"' in evaluation_block
    assert 'RESEARCH_CPU_BUDGET: "24"' in evaluation_block
    assert 'RESEARCH_MEMORY_BUDGET_GB: "40"' in evaluation_block
    assert "strategy_health_collect" in evaluation_block


def test_strategy_health_private_materialization_is_recent_only() -> None:
    source = (
        Path(__file__).parents[1] / "scripts" / "materialize_factor_library.py"
    ).read_text(encoding="utf-8")

    assert 'str(feature_set["id"]).startswith("strategy-health:")' in source
    assert '"recent_only"' in source
    assert 'if storage_mode == "full_and_recent":' in source

    collector = (
        Path(__file__).parents[1] / "scripts" / "collect_strategy_health.py"
    ).read_text(encoding="utf-8")
    assert "observed_at=datetime.now(UTC)" in collector
    assert "args.observed_at" not in collector
