"""Design-gap fixes: sealed OOS vintages (4.1/12.1) and failed-trial ledgering (4.2/6.6)."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import pytest
from governance_fixtures import (
    DATASET_IDENTITY,
    PERIODS,
    create_promoted_factor,
    create_strategy_version,
    passing_factor_metrics,
)
from qlib_test_doubles import qlib_workflow_identity
from sqlalchemy import select

from quant_data.config import Settings
from quant_data.database import oos_vintages, open_database, row_dict
from quant_platform.api import StrategyConfigRequest
from quant_platform.cost_model import CostModelConfig
from quant_platform.job_store import JobStore
from quant_platform.research_automation import rank_factor_candidates
from quant_platform.research_campaign_store import ResearchCampaignStore
from quant_platform.research_program_store import ResearchProgramStore
from quant_platform.research_store import ResearchStore
from quant_platform.strategy_store import StrategyStore
from quant_platform.worker import LocalJobWorker

FINAL_PERIODS = {
    "start": PERIODS["test_start"].isoformat(),
    "end": PERIODS["test_end"].isoformat(),
}


def _vintage_rows(database_url: str) -> list[dict]:
    engine = open_database(database_url)
    with engine.connect() as connection:
        return [row_dict(row) for row in connection.execute(select(oos_vintages)).all()]


def _version_for_candidate(database_url: str, candidate_id: str) -> str:
    config = {
        "topk": 50,
        "n_drop": 5,
        "max_tracking_error": 0.12,
        "max_drawdown": 0.25,
        "max_turnover": 0.60,
        "min_information_ratio": 0.0,
        "min_sharpe_ratio": 0.0,
        "min_sortino_ratio": 0.0,
        "min_rolling_pass_rate": 0.60,
        "min_rolling_windows": 3,
        "event_count": 5,
        "min_backtest_days": 504,
        "capacity_notional": 5_000_000,
        "max_volume_participation": 0.01,
        "min_commission": 5.0,
    }
    config.update(CostModelConfig().to_dict())
    config = StrategyConfigRequest.model_validate(config).model_dump()
    strategy = StrategyStore(database_url).create(
        name=f"strategy-{uuid.uuid4().hex}",
        description="OOS vintage test strategy.",
        benchmark="SH000300",
        universe="cn_all",
        factors=[{"candidate_id": candidate_id, "weight": 1.0}],
        config=config,
        actor="test",
    )
    return str(strategy["versions"][0]["id"])


def _record_passing_evaluation(store: ResearchStore, tmp_path: Path, candidate: dict) -> dict:
    suffix = uuid.uuid4().hex
    metrics = passing_factor_metrics()
    artifact = tmp_path / f"evaluation-{suffix}.json"
    artifact.write_text(
        json.dumps(
            {
                "status": "ok",
                "qlib_workflow": qlib_workflow_identity(),
                "evaluations": [
                    {"candidate_id": candidate["id"], "status": "ok", "metrics": metrics}
                ],
            }
        ),
        encoding="utf-8",
    )
    recomputed_path = tmp_path / f"recomputed-{suffix}.h5"
    recomputed_path.write_bytes(b"independently-recomputed-factor-values")
    recomputed_sha256 = hashlib.sha256(recomputed_path.read_bytes()).hexdigest()
    return store.record_evaluation(
        candidate["id"],
        dataset="snapshot",
        dataset_identity_sha256=DATASET_IDENTITY,
        **PERIODS,
        metrics=metrics,
        artifact_path=str(artifact),
        recomputed_values_path=str(recomputed_path),
        recomputed_values_sha256=recomputed_sha256,
        recompute_evidence={
            "executor_version": "factor-recompute-v3-container-index-exact",
            "sandbox_mode": "docker-isolated",
            "sandbox_image_id": "sha256:" + "a" * 64,
            "network_mode": "none",
            "root_filesystem_read_only": True,
            "capabilities_dropped": "ALL",
            "no_new_privileges": True,
            "label_horizon_days": 1,
            "code_sha256": hashlib.sha256(Path(candidate["code_path"]).read_bytes()).hexdigest(),
            "dataset_identity_sha256": DATASET_IDENTITY,
            "provider_input_sha256": "1" * 64,
            "periods": {key: value.isoformat() for key, value in PERIODS.items()},
            "submitted_comparison": {
                "available": True,
                "exact_match": True,
                "index_exact_match": True,
                "submitted_sha256": hashlib.sha256(
                    Path(candidate["values_path"]).read_bytes()
                ).hexdigest(),
            },
            "authoritative_values_sha256": recomputed_sha256,
        },
    )


def _unevaluated_candidate(
    store: ResearchStore, tmp_path: Path, run: dict | None = None
) -> tuple[dict, dict]:
    suffix = uuid.uuid4().hex
    if run is None:
        run = store.create_run(
            kind=f"factor-{suffix}",
            objective="Create unevaluated candidate fixture.",
            dataset="snapshot",
            requested_by="test",
            budget={"loop_n": 1},
            config={},
            artifact_path=tmp_path,
        )
    code_path = tmp_path / f"factor-{suffix}.py"
    values_path = tmp_path / f"factor-{suffix}.h5"
    code_path.write_text("def factor(frame):\n    return frame['close']\n", encoding="utf-8")
    values_path.write_bytes(b"immutable-factor-values")
    candidate = store.add_candidate(
        run["id"],
        name=f"factor-{suffix}",
        description="unevaluated fixture",
        formulation="close",
        variables={},
        source_iteration=0,
        code_path=str(code_path),
        values_path=str(values_path),
        code_sha256=hashlib.sha256(code_path.read_bytes()).hexdigest(),
        rdagent_decision=True,
        rdagent_feedback="ok",
    )
    return run, candidate


def test_first_final_test_seals_and_consumes_vintage(
    tmp_path: Path, database_url: str
) -> None:
    version_id = create_strategy_version(database_url, tmp_path)
    store = StrategyStore(database_url)
    version = store.get_version(version_id)
    candidate_id = str(version["factors"][0]["factor_candidate_id"])
    store.create_backtest(
        version_id=version_id,
        dataset="snapshot",
        periods=FINAL_PERIODS,
        artifact_path=tmp_path,
    )
    rows = _vintage_rows(database_url)
    assert len(rows) == 1
    row = rows[0]
    assert row["scope"] == f"dataset:{DATASET_IDENTITY}"
    assert row["dataset_identity"] == DATASET_IDENTITY
    assert row["test_start"] == PERIODS["test_start"]
    assert row["test_end"] == PERIODS["test_end"]
    assert row["sealed_at"] is not None
    assert row["first_opened_at"] is not None
    assert row["consumed_at"] is not None
    assert row["sealed_candidate_set_json"]["candidate_ids"] == [candidate_id]
    assert len(row["sealed_candidate_set_sha256"]) == 64


def test_new_candidate_cannot_reconsume_sealed_window(
    tmp_path: Path, database_url: str
) -> None:
    """A new evaluation row (new run / renamed campaign) may not reopen the window."""

    version_a = create_strategy_version(database_url, tmp_path)
    store = StrategyStore(database_url)
    store.create_backtest(
        version_id=version_a, dataset="snapshot", periods=FINAL_PERIODS, artifact_path=tmp_path
    )
    version_b = create_strategy_version(database_url, tmp_path)
    with pytest.raises(ValueError, match="sealed candidate set"):
        store.create_backtest(
            version_id=version_b,
            dataset="snapshot",
            periods=FINAL_PERIODS,
            artifact_path=tmp_path,
        )


def test_same_candidate_new_version_rejected_after_consumption(
    tmp_path: Path, database_url: str
) -> None:
    """The OOS is one-time even for candidates already in the sealed set."""

    version_id = create_strategy_version(database_url, tmp_path)
    store = StrategyStore(database_url)
    store.create_backtest(
        version_id=version_id, dataset="snapshot", periods=FINAL_PERIODS, artifact_path=tmp_path
    )
    version = store.get_version(version_id)
    frozen = store.create_version(
        str(version["strategy_id"]),
        benchmark=version["benchmark"],
        universe=version["universe"],
        factors=[
            {"candidate_id": item["factor_candidate_id"], "weight": item["weight"]}
            for item in version["factors"]
        ],
        config=version["config"],
        actor="test",
    )
    with pytest.raises(ValueError, match="already been consumed"):
        store.create_backtest(
            version_id=frozen["id"],
            dataset="snapshot",
            periods=FINAL_PERIODS,
            artifact_path=tmp_path,
        )


def test_program_scope_binds_vintage_across_renamed_campaigns(
    tmp_path: Path, database_url: str
) -> None:
    """A second campaign (new name/lineage) under the same program cannot reopen."""

    programs = ResearchProgramStore(database_url)
    campaigns = ResearchCampaignStore(database_url)
    program = programs.create(
        name=f"program-{uuid.uuid4().hex}",
        recipe_id="recipe",
        objective="governed research",
        benchmark="SH000300",
        universe="cn_all",
        dataset_lineage_id="lineage",
        config={},
        min_new_trading_days=1,
        max_active_campaigns=2,
        actor="test",
    )
    candidate_a = create_promoted_factor(database_url, tmp_path)
    campaign_a = campaigns.create(
        name=f"campaign-alpha-{uuid.uuid4().hex}",
        objective="first campaign",
        dataset="snapshot",
        benchmark="SH000300",
        universe="cn_all",
        recipe_id="recipe",
        config={},
        actor="test",
        research_program_id=program["id"],
        dataset_identity_sha256=DATASET_IDENTITY,
    )
    campaigns.transition(
        campaign_a["id"],
        event_type="test.link",
        links={"research_run_id": str(candidate_a["research_run_id"])},
    )
    version_a = _version_for_candidate(database_url, str(candidate_a["id"]))
    strategies = StrategyStore(database_url)
    strategies.create_backtest(
        version_id=version_a, dataset="snapshot", periods=FINAL_PERIODS, artifact_path=tmp_path
    )
    rows = _vintage_rows(database_url)
    assert len(rows) == 1
    assert rows[0]["scope"] == f"program:{program['id']}"

    candidate_b = create_promoted_factor(database_url, tmp_path)
    campaign_b = campaigns.create(
        name=f"campaign-beta-{uuid.uuid4().hex}",
        objective="renamed successor campaign",
        dataset="snapshot",
        benchmark="SH000300",
        universe="cn_all",
        recipe_id="recipe",
        config={},
        actor="test",
        research_program_id=program["id"],
        dataset_identity_sha256="b" * 64,
    )
    campaigns.transition(
        campaign_b["id"],
        event_type="test.link",
        links={"research_run_id": str(candidate_b["research_run_id"])},
    )
    version_b = _version_for_candidate(database_url, str(candidate_b["id"]))
    with pytest.raises(ValueError, match="sealed candidate set"):
        strategies.create_backtest(
            version_id=version_b,
            dataset="snapshot",
            periods=FINAL_PERIODS,
            artifact_path=tmp_path,
        )
    assert len(_vintage_rows(database_url)) == 1


def test_failed_evaluation_is_ledgered_and_blocks_promotion(
    tmp_path: Path, database_url: str
) -> None:
    store = ResearchStore(database_url)
    _, candidate = _unevaluated_candidate(store, tmp_path)
    failure = store.record_failed_evaluation(
        candidate["id"],
        dataset="snapshot",
        dataset_identity_sha256=DATASET_IDENTITY,
        **PERIODS,
        error="qlib process exploded",
    )
    assert failure["gate_status"] == "evaluation_failed"
    assert failure["gate_reasons"] == ["qlib process exploded"]
    assert failure["metrics"] == {}
    run_context_sha256 = failure["recompute_evidence"]["run_context_sha256"]
    assert len(run_context_sha256) == 64
    state = store.get_candidate(candidate["id"])
    assert state["status"] == "evaluation_failed"
    assert state["latest_evaluation"]["id"] == failure["id"]
    # evaluation_failed is unqualified everywhere downstream: the promotion
    # gate rejects it and deterministic ranking excludes it (it is not the
    # same as gate_failed, where the evaluation completed but missed the bar).
    with pytest.raises(ValueError, match="must pass the latest Qlib gate"):
        store.promote(candidate["id"], actor="risk-owner", reason="attempting promotion")
    assert rank_factor_candidates([store.get_candidate(candidate["id"])], limit=1) == []


def test_failed_candidate_can_be_reevaluated_and_promoted(
    tmp_path: Path, database_url: str
) -> None:
    """The failure row stays in the ledger; a later success still promotes."""

    store = ResearchStore(database_url)
    _, candidate = _unevaluated_candidate(store, tmp_path)
    failure = store.record_failed_evaluation(
        candidate["id"],
        dataset="snapshot",
        dataset_identity_sha256=DATASET_IDENTITY,
        **PERIODS,
        error="transient qlib crash",
    )
    passed = _record_passing_evaluation(store, tmp_path, candidate)
    state = store.get_candidate(candidate["id"])
    assert state["status"] == "gate_passed"
    assert state["latest_evaluation"]["id"] == passed["id"]
    promoted = store.promote(
        candidate["id"], actor="factor-owner", reason="Re-evaluation passed the gate."
    )
    assert promoted["status"] == "promoted"
    evaluations = [
        event
        for event in store.list_events(candidate["research_run_id"])
        if event["event_type"] == "candidate.evaluation_failed"
    ]
    assert evaluations and evaluations[0]["payload"]["evaluation_id"] == failure["id"]


def test_worker_import_ledgers_failed_evaluations(
    tmp_path: Path, database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("RUN_EMBEDDED_WORKER", "false")
    settings = Settings.from_env(tmp_path / ".env")
    worker = LocalJobWorker(JobStore(database_url), tmp_path, settings)
    store = ResearchStore(database_url)
    run, candidate = _unevaluated_candidate(store, tmp_path)
    job = {
        "payload": {
            "research_run_id": run["id"],
            "dataset": "snapshot",
            "dataset_identity_sha256": DATASET_IDENTITY,
            "periods": {key: value.isoformat() for key, value in PERIODS.items()},
        }
    }
    result = {
        "evaluations": [
            {"candidate_id": candidate["id"], "status": "failed", "error": "recompute timeout"}
        ]
    }
    worker._import_factor_evaluations(job, result)
    state = store.get_candidate(candidate["id"])
    assert state["status"] == "evaluation_failed"
    assert state["latest_evaluation"]["gate_status"] == "evaluation_failed"
    assert state["latest_evaluation"]["gate_reasons"] == ["recompute timeout"]
