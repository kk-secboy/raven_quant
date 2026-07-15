import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from quant_platform.research_store import ResearchStore

PERIODS = {
    "train_start": date(2018, 1, 1),
    "train_end": date(2021, 12, 31),
    "valid_start": date(2022, 1, 1),
    "valid_end": date(2023, 12, 31),
    "test_start": date(2024, 1, 1),
    "test_end": date(2026, 7, 10),
}
DATASET_IDENTITY = "a" * 64


def _passing_metrics() -> dict:
    return {
        "ic": 0.035,
        "icir": 0.80,
        "rank_ic": 0.041,
        "rank_icir": 0.76,
        "turnover": 0.32,
        "max_correlation": 0.44,
        "cost_adjusted_return": 0.052,
        "raw_valid_ic": -0.031,
        "raw_selection_ic": -0.035,
        "selection_days": 400,
        "selection_start": "2022-05-27",
        "coverage_pass_rate": 0.99,
        "mean_coverage_ratio": 0.95,
        "constant_day_rate": 0.0,
        "direction": "inverted",
    }


def _candidate(store: ResearchStore, tmp_path: Path) -> dict:
    run = store.create_run(
        kind="factor",
        objective="Find a low-turnover quality factor for CSI 300 enhancement.",
        dataset="snapshot-20260710",
        requested_by="researcher",
        budget={"loop_n": 1, "duration": "30m"},
        config={"periods": {key: value.isoformat() for key, value in PERIODS.items()}},
        artifact_path=tmp_path,
    )
    code_path = tmp_path / "factor.py"
    values_path = tmp_path / "factor.h5"
    code_path.write_text("def factor(frame):\n    return frame['close']\n", encoding="utf-8")
    values_path.write_bytes(b"immutable-factor-values")
    return store.add_candidate(
        run["id"],
        name="quality_stability",
        description="Stable profitability with conservative balance-sheet growth.",
        formulation="zscore(roe_ttm) - zscore(asset_growth)",
        variables={"roe_ttm": "ROE", "asset_growth": "asset growth"},
        source_iteration=0,
        code_path=str(code_path),
        values_path=str(values_path),
        code_sha256=hashlib.sha256(code_path.read_bytes()).hexdigest(),
        rdagent_decision=True,
        rdagent_feedback="implementation passed",
    )


def _write_evaluation_artifact(path: Path, candidate_id: str, metrics: dict) -> Path:
    path.write_text(
        json.dumps(
            {
                "status": "ok",
                "evaluations": [{"candidate_id": candidate_id, "status": "ok", "metrics": metrics}],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _recompute_args(store: ResearchStore, candidate_id: str, tmp_path: Path) -> dict:
    candidate = store.get_candidate(candidate_id)
    sequence = len(list(tmp_path.glob("recomputed-*")))
    recomputed_path = tmp_path / f"recomputed-{candidate_id}-{sequence}.h5"
    recomputed_path.write_bytes(b"independently-recomputed-factor-values")
    recomputed_sha256 = hashlib.sha256(recomputed_path.read_bytes()).hexdigest()
    return {
        "recomputed_values_path": str(recomputed_path),
        "recomputed_values_sha256": recomputed_sha256,
        "recompute_evidence": {
            "executor_version": "factor-recompute-v1",
            "code_sha256": candidate["code_sha256"],
            "dataset_identity_sha256": DATASET_IDENTITY,
            "provider_input_sha256": "1" * 64,
            "periods": {key: value.isoformat() for key, value in PERIODS.items()},
            "submitted_comparison": {
                "available": True,
                "exact_match": True,
                "submitted_sha256": candidate["values_sha256"],
            },
            "authoritative_values_sha256": recomputed_sha256,
        },
    }


def test_factor_must_pass_qlib_gate_before_manual_promotion(
    tmp_path: Path, database_url: str
) -> None:
    store = ResearchStore(database_url)
    candidate = _candidate(store, tmp_path)

    failed = store.record_evaluation(
        candidate["id"],
        dataset="snapshot-20260710",
        dataset_identity_sha256=DATASET_IDENTITY,
        **PERIODS,
        metrics={
            "ic": 0.01,
            "icir": 0.20,
            "rank_ic": 0.01,
            "rank_icir": 0.20,
            "turnover": 0.80,
            "max_correlation": 0.90,
            "cost_adjusted_return": -0.01,
            "valid_ic": 0.03,
            "test_days": 500,
        },
        artifact_path=None,
        **_recompute_args(store, candidate["id"], tmp_path),
    )
    assert failed["gate_status"] == "failed"
    assert store.get_candidate(candidate["id"])["status"] == "gate_failed"
    with pytest.raises(ValueError, match="must pass"):
        store.promote(candidate["id"], actor="portfolio-owner", reason="not ready yet")

    passed_metrics = _passing_metrics()
    evaluation_artifact = _write_evaluation_artifact(
        tmp_path / "evaluation.json", candidate["id"], passed_metrics
    )
    passed = store.record_evaluation(
        candidate["id"],
        dataset="snapshot-20260710",
        dataset_identity_sha256=DATASET_IDENTITY,
        **PERIODS,
        metrics=passed_metrics,
        artifact_path=str(evaluation_artifact),
        **_recompute_args(store, candidate["id"], tmp_path),
    )
    assert passed["gate_status"] == "passed"
    assert len(passed["evidence_sha256"]) == 64
    promoted = store.promote(
        candidate["id"],
        actor="portfolio-owner",
        reason="Passed independent out-of-sample and cost-aware validation.",
    )
    assert promoted["status"] == "promoted"
    assert promoted["promoted_evaluation_id"] == passed["id"]
    assert promoted["promotion_evidence_sha256"] == passed["evidence_sha256"]
    events = store.list_events(candidate["research_run_id"])
    assert {event["event_type"] for event in events} >= {
        "candidate.gate_failed",
        "candidate.gate_passed",
        "candidate.promoted",
    }


@pytest.mark.parametrize("artifact_name", ["factor.py", "recomputed", "evaluation.json"])
def test_factor_promotion_rejects_evidence_changed_after_evaluation(
    tmp_path: Path, database_url: str, artifact_name: str
) -> None:
    store = ResearchStore(database_url)
    candidate = _candidate(store, tmp_path)
    metrics = _passing_metrics()
    artifact = _write_evaluation_artifact(tmp_path / "evaluation.json", candidate["id"], metrics)
    recompute_args = _recompute_args(store, candidate["id"], tmp_path)
    store.record_evaluation(
        candidate["id"],
        dataset="snapshot-20260710",
        dataset_identity_sha256=DATASET_IDENTITY,
        **PERIODS,
        metrics=metrics,
        artifact_path=str(artifact),
        **recompute_args,
    )
    target = (
        Path(recompute_args["recomputed_values_path"])
        if artifact_name == "recomputed"
        else tmp_path / artifact_name
    )
    target.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="changed"):
        store.promote(
            candidate["id"],
            actor="portfolio-owner",
            reason="This evidence must remain immutable before promotion.",
        )


def test_factor_gate_fails_closed_when_metrics_are_missing(
    tmp_path: Path, database_url: str
) -> None:
    store = ResearchStore(database_url)
    candidate = _candidate(store, tmp_path)
    evaluation = store.record_evaluation(
        candidate["id"],
        dataset="snapshot-20260710",
        dataset_identity_sha256=DATASET_IDENTITY,
        **PERIODS,
        metrics={"ic": 0.04},
        artifact_path=None,
        **_recompute_args(store, candidate["id"], tmp_path),
    )
    assert evaluation["gate_status"] == "failed"
    assert any("is missing" in reason for reason in evaluation["gate_reasons"])


def test_factor_evaluation_rejects_submitted_values_that_do_not_match_recompute(
    tmp_path: Path, database_url: str
) -> None:
    store = ResearchStore(database_url)
    candidate = _candidate(store, tmp_path)
    recompute = _recompute_args(store, candidate["id"], tmp_path)
    recompute["recompute_evidence"]["submitted_comparison"]["exact_match"] = False

    with pytest.raises(ValueError, match="do not match independent recomputation"):
        store.record_evaluation(
            candidate["id"],
            dataset="snapshot-20260710",
            dataset_identity_sha256=DATASET_IDENTITY,
            **PERIODS,
            metrics=_passing_metrics(),
            artifact_path=str(
                _write_evaluation_artifact(
                    tmp_path / "mismatched-evaluation.json",
                    candidate["id"],
                    _passing_metrics(),
                )
            ),
            **recompute,
        )


def test_only_one_active_factor_research_pipeline_is_allowed(
    tmp_path: Path, database_url: str
) -> None:
    store = ResearchStore(database_url)
    store.create_run(
        kind="factor",
        objective="First bounded factor research pipeline.",
        dataset="snapshot-a",
        requested_by="researcher",
        budget={"loop_n": 1, "duration": "30m"},
        config={},
        artifact_path=tmp_path,
    )
    with pytest.raises(ValueError, match="active factor research run"):
        store.create_run(
            kind="factor",
            objective="Second overlapping factor research pipeline.",
            dataset="snapshot-b",
            requested_by="researcher",
            budget={"loop_n": 1, "duration": "30m"},
            config={},
            artifact_path=tmp_path,
        )
