from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from qlib_test_doubles import qlib_workflow_identity

from quant_platform import external_factor_evaluation as ext
from quant_platform.factor_evaluator import evaluate_factor_values
from quant_platform.research_store import (
    EXTERNAL_EVALUATOR_VERSION,
    ExternalEventGatePolicy,
    MarketTimeseriesGatePolicy,
    ResearchStore,
)

DAYS = pd.bdate_range("2022-01-04", periods=140)
INSTRUMENTS = [f"6000{i:02d}.SH" for i in range(20)]
BENCHMARK = "SH000300"

PERIODS = {
    "train_start": date(2018, 1, 1),
    "train_end": date(2021, 12, 31),
    "valid_start": DAYS[0].date(),
    "valid_end": DAYS[-1].date(),
    "test_start": DAYS[-1].date() + timedelta(days=10),
    "test_end": DAYS[-1].date() + timedelta(days=370),
}
DATASET_IDENTITY = "b" * 64
INPUT_DATA_SHA256 = "c" * 64

CONSISTENCY_KEYS = (
    "ic",
    "icir",
    "rank_ic",
    "rank_icir",
    "hac_p_value",
    "turnover",
    "cost_adjusted_return",
    "gross_annualized_return",
    "direction",
    "selection_days",
    "observations",
    "raw_valid_ic",
    "raw_selection_ic",
)


def _series(rows: list[tuple], name: str) -> pd.Series:
    frame = pd.DataFrame(rows, columns=["datetime", "instrument", name])
    return frame.set_index(["datetime", "instrument"])[name].sort_index()


def _sparse_event_inputs(
    *,
    event_days: int,
    instruments_per_day: int = 12,
    effect: float = 0.02,
    noise: float = 0.01,
    seed: int = 11,
) -> tuple[pd.Series, pd.Series]:
    """Sparse event factor plus dense universe forward returns.

    Signals are persistent per instrument (plus small daily jitter) so the
    event long/short basket is stable enough for the turnover gate; forward
    returns carry the signal with the given effect strength.
    """

    rng = np.random.default_rng(seed)
    chosen_days = list(DAYS[::2][:event_days])
    base_signal = {instrument: float(rng.standard_normal()) for instrument in INSTRUMENTS}
    event_instruments = INSTRUMENTS[:instruments_per_day]
    signals: dict[tuple[pd.Timestamp, str], float] = {}
    factor_rows = []
    for day in chosen_days:
        for instrument in event_instruments:
            signal = base_signal[instrument] + 0.1 * float(rng.standard_normal())
            signals[(day, instrument)] = signal
            factor_rows.append((day, instrument, signal))
    label_rows = []
    for day in DAYS:
        for instrument in INSTRUMENTS:
            value = noise * float(rng.standard_normal())
            signal = signals.get((day, instrument))
            if signal is not None:
                value += effect * signal
            label_rows.append((day, instrument, value))
    return _series(factor_rows, "factor"), _series(label_rows, "label")


def _market_inputs(
    *,
    days: int = 140,
    effect: float = 0.02,
    noise: float = 0.01,
    seed: int = 23,
) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    used_days = DAYS[:days]
    signal = rng.standard_normal(len(used_days))
    returns = effect * signal + noise * rng.standard_normal(len(used_days))
    factor = _series(
        [
            (day, ext.MARKET_INSTRUMENT, float(value))
            for day, value in zip(used_days, signal, strict=True)
        ],
        "factor",
    )
    label = _series(
        [(day, BENCHMARK, float(value)) for day, value in zip(used_days, returns, strict=True)],
        "label",
    )
    return factor, label


def _event_entry(candidate_id: str, metrics: dict, *, family: str = "fam", count: int = 1) -> dict:
    entry = {
        "candidate_id": candidate_id,
        "status": "ok",
        "metrics": metrics,
        "experiment_family_id": family,
        "experiment_count": count,
    }
    ext.apply_family_bh_correction([entry])
    return entry


def _write_result_artifact(path: Path, candidate_id: str, metrics: dict) -> Path:
    path.write_text(
        json.dumps(
            {
                "status": "ok",
                "qlib_workflow": qlib_workflow_identity(),
                "evaluations": [{"candidate_id": candidate_id, "status": "ok", "metrics": metrics}],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Shape detection (fail-closed)
# ---------------------------------------------------------------------------


@pytest.mark.no_database
def test_shape_detection_sparse_event_and_hint_contract() -> None:
    factor, label = _sparse_event_inputs(event_days=40)
    shape = ext.detect_external_factor_shape(
        factor, label, valid_start=PERIODS["valid_start"], valid_end=PERIODS["valid_end"]
    )
    assert shape == ext.SHAPE_SPARSE_EVENT
    hinted = ext.detect_external_factor_shape(
        factor,
        label,
        valid_start=PERIODS["valid_start"],
        valid_end=PERIODS["valid_end"],
        shape_hint="sparse_event",
    )
    assert hinted == ext.SHAPE_SPARSE_EVENT
    with pytest.raises(ValueError, match="contradicts"):
        ext.detect_external_factor_shape(
            factor,
            label,
            valid_start=PERIODS["valid_start"],
            valid_end=PERIODS["valid_end"],
            shape_hint="market_timeseries",
        )
    with pytest.raises(ValueError, match="unsupported external factor shape hint"):
        ext.detect_external_factor_shape(
            factor,
            label,
            valid_start=PERIODS["valid_start"],
            valid_end=PERIODS["valid_end"],
            shape_hint="cross_sectional",
        )


@pytest.mark.no_database
def test_shape_detection_market_timeseries() -> None:
    factor, label = _market_inputs()
    shape = ext.detect_external_factor_shape(
        factor, label, valid_start=PERIODS["valid_start"], valid_end=PERIODS["valid_end"]
    )
    assert shape == ext.SHAPE_MARKET_TIMESERIES


@pytest.mark.no_database
def test_shape_detection_fails_closed_on_dense_cross_section() -> None:
    rng = np.random.default_rng(3)
    rows = [
        (day, instrument, float(rng.standard_normal()))
        for day in DAYS
        for instrument in INSTRUMENTS
    ]
    _, label = _sparse_event_inputs(event_days=40)
    with pytest.raises(ValueError, match="cross-sectionally dense"):
        ext.detect_external_factor_shape(
            _series(rows, "factor"),
            label,
            valid_start=PERIODS["valid_start"],
            valid_end=PERIODS["valid_end"],
        )


@pytest.mark.no_database
def test_shape_detection_fails_closed_on_ambiguous_and_mixed_shapes() -> None:
    factor, label = _sparse_event_inputs(event_days=40)
    rng = np.random.default_rng(5)
    ambiguous_rows = [
        (day, instrument, float(rng.standard_normal()))
        for day in DAYS[:100]
        for instrument in INSTRUMENTS[:3]
    ]
    with pytest.raises(ValueError, match="neither event-sparse nor dense"):
        ext.detect_external_factor_shape(
            _series(ambiguous_rows, "factor"),
            label,
            valid_start=PERIODS["valid_start"],
            valid_end=PERIODS["valid_end"],
        )
    market_rows = [
        (day, ext.MARKET_INSTRUMENT, 0.1) for day in DAYS[:40]
    ]
    mixed = pd.concat([factor, _series(market_rows, "factor")])
    with pytest.raises(ValueError, match="mixes the MARKET pseudo-instrument"):
        ext.detect_external_factor_shape(
            mixed,
            label,
            valid_start=PERIODS["valid_start"],
            valid_end=PERIODS["valid_end"],
        )
    with pytest.raises(ValueError, match="empty in the validation window"):
        ext.detect_external_factor_shape(
            factor.head(0),
            label,
            valid_start=PERIODS["valid_start"],
            valid_end=PERIODS["valid_end"],
        )


# ---------------------------------------------------------------------------
# Sparse event evaluation path
# ---------------------------------------------------------------------------


@pytest.mark.no_database
def test_sparse_event_evaluation_passes_event_gate() -> None:
    factor, label = _sparse_event_inputs(event_days=40)
    outcome = ext.evaluate_sparse_event_factor(
        factor,
        label,
        valid_start=PERIODS["valid_start"],
        valid_end=PERIODS["valid_end"],
        test_start=PERIODS["test_start"],
        test_end=PERIODS["test_end"],
    )
    assert outcome["status"] == "ok"
    metrics = outcome["metrics"]
    assert metrics["evaluation_shape"] == ext.SHAPE_SPARSE_EVENT
    assert metrics["event_days"] == 40
    assert metrics["selection_days"] == 32
    assert metrics["ic"] > 0.3
    assert metrics["external_evaluator_version"] == EXTERNAL_EVALUATOR_VERSION
    entry = _event_entry("candidate-1", metrics)
    gate_status, reasons = ExternalEventGatePolicy().evaluate(entry["metrics"])
    assert gate_status == "passed", reasons


@pytest.mark.no_database
def test_sparse_event_evaluation_matches_evaluate_factor_values() -> None:
    factor, label = _sparse_event_inputs(event_days=40)
    outcome = ext.evaluate_sparse_event_factor(
        factor,
        label,
        valid_start=PERIODS["valid_start"],
        valid_end=PERIODS["valid_end"],
        test_start=PERIODS["test_start"],
        test_end=PERIODS["test_end"],
    )
    direct = evaluate_factor_values(
        factor,
        label,
        valid_start=PERIODS["valid_start"],
        valid_end=PERIODS["valid_end"],
        test_start=PERIODS["test_start"],
        test_end=PERIODS["test_end"],
        min_daily_instruments=5,
        label_horizon_days=1,
    )
    assert outcome["status"] == "ok"
    for key in CONSISTENCY_KEYS:
        assert outcome["metrics"][key] == direct[key], key


@pytest.mark.no_database
def test_sparse_event_evaluation_insufficient_when_events_too_few() -> None:
    factor, label = _sparse_event_inputs(event_days=8)
    outcome = ext.evaluate_sparse_event_factor(
        factor,
        label,
        valid_start=PERIODS["valid_start"],
        valid_end=PERIODS["valid_end"],
        test_start=PERIODS["test_start"],
        test_end=PERIODS["test_end"],
    )
    assert outcome["status"] == "insufficient_evidence"
    assert outcome["metrics"] is None
    assert any("event days=8" in reason for reason in outcome["reasons"])


@pytest.mark.no_database
def test_sparse_event_gate_reports_insufficient_below_event_floor() -> None:
    factor, label = _sparse_event_inputs(event_days=20)
    outcome = ext.evaluate_sparse_event_factor(
        factor,
        label,
        valid_start=PERIODS["valid_start"],
        valid_end=PERIODS["valid_end"],
        test_start=PERIODS["test_start"],
        test_end=PERIODS["test_end"],
    )
    assert outcome["status"] == "ok"
    assert outcome["metrics"]["selection_days"] == 16
    entry = _event_entry("candidate-1", outcome["metrics"])
    gate_status, reasons = ExternalEventGatePolicy().evaluate(entry["metrics"])
    assert gate_status == "insufficient_evidence"
    assert any("event days=16" in reason for reason in reasons)


@pytest.mark.no_database
def test_sparse_event_gate_fails_on_weak_signal_without_relaxing() -> None:
    factor, label = _sparse_event_inputs(event_days=40, effect=0.0)
    outcome = ext.evaluate_sparse_event_factor(
        factor,
        label,
        valid_start=PERIODS["valid_start"],
        valid_end=PERIODS["valid_end"],
        test_start=PERIODS["test_start"],
        test_end=PERIODS["test_end"],
    )
    assert outcome["status"] == "ok"
    entry = _event_entry("candidate-1", outcome["metrics"])
    gate_status, reasons = ExternalEventGatePolicy().evaluate(entry["metrics"])
    assert gate_status == "failed"
    assert reasons


# ---------------------------------------------------------------------------
# Market timeseries evaluation path
# ---------------------------------------------------------------------------


@pytest.mark.no_database
def test_market_timeseries_evaluation_passes_market_gate() -> None:
    factor, label = _market_inputs()
    outcome = ext.evaluate_market_timeseries_factor(
        factor,
        label,
        valid_start=PERIODS["valid_start"],
        valid_end=PERIODS["valid_end"],
        test_start=PERIODS["test_start"],
        test_end=PERIODS["test_end"],
    )
    assert outcome["status"] == "ok"
    metrics = outcome["metrics"]
    assert metrics["evaluation_shape"] == ext.SHAPE_MARKET_TIMESERIES
    assert metrics["benchmark_instrument"] == BENCHMARK
    assert metrics["selection_days"] == 112
    assert metrics["ic"] > 0.3
    assert metrics["quantile_spread"] > 0
    entry = _event_entry("candidate-1", metrics)
    gate_status, reasons = MarketTimeseriesGatePolicy().evaluate(entry["metrics"])
    assert gate_status == "passed", reasons


@pytest.mark.no_database
def test_market_timeseries_evaluation_insufficient_when_days_too_few() -> None:
    factor, label = _market_inputs(days=8)
    outcome = ext.evaluate_market_timeseries_factor(
        factor,
        label,
        valid_start=PERIODS["valid_start"],
        valid_end=PERIODS["valid_end"],
        test_start=PERIODS["test_start"],
        test_end=PERIODS["test_end"],
    )
    assert outcome["status"] == "insufficient_evidence"
    assert outcome["metrics"] is None
    assert any("signal days=8" in reason for reason in outcome["reasons"])


@pytest.mark.no_database
def test_market_timeseries_gate_reports_insufficient_below_signal_floor() -> None:
    factor, label = _market_inputs(days=60)
    outcome = ext.evaluate_market_timeseries_factor(
        factor,
        label,
        valid_start=PERIODS["valid_start"],
        valid_end=PERIODS["valid_end"],
        test_start=PERIODS["test_start"],
        test_end=PERIODS["test_end"],
    )
    assert outcome["status"] == "ok"
    assert outcome["metrics"]["selection_days"] == 48
    entry = _event_entry("candidate-1", outcome["metrics"])
    gate_status, reasons = MarketTimeseriesGatePolicy().evaluate(entry["metrics"])
    assert gate_status == "insufficient_evidence"
    assert any("signal days=48" in reason for reason in reasons)


@pytest.mark.no_database
def test_market_timeseries_evaluation_fails_closed_on_wrong_inputs() -> None:
    factor, label = _market_inputs()
    not_market = _series([(day, "600000.SH", 0.1) for day in DAYS[:40]], "factor")
    with pytest.raises(ValueError, match="MARKET pseudo-instrument"):
        ext.evaluate_market_timeseries_factor(
            not_market,
            label,
            valid_start=PERIODS["valid_start"],
            valid_end=PERIODS["valid_end"],
            test_start=PERIODS["test_start"],
            test_end=PERIODS["test_end"],
        )
    two_benchmarks = pd.concat(
        [label, _series([(day, "SH000905", 0.01) for day in DAYS], "label")]
    )
    with pytest.raises(ValueError, match="exactly one instrument"):
        ext.evaluate_market_timeseries_factor(
            factor,
            two_benchmarks,
            valid_start=PERIODS["valid_start"],
            valid_end=PERIODS["valid_end"],
            test_start=PERIODS["test_start"],
            test_end=PERIODS["test_end"],
        )


@pytest.mark.no_database
def test_family_bh_correction_pads_declared_experiment_count() -> None:
    first = {"candidate_id": "a", "status": "ok", "metrics": {"hac_p_value": 0.01},
             "experiment_family_id": "fam", "experiment_count": 4}
    second = {"candidate_id": "b", "status": "insufficient_evidence", "metrics": None,
              "experiment_family_id": "fam", "experiment_count": 4}
    ext.apply_family_bh_correction([first, second])
    # p-values [0.01, 1.0] padded to the declared 4 experiments -> q = 0.01 * 4 / 1.
    assert first["metrics"]["bh_q_value"] == pytest.approx(0.04)
    assert first["metrics"]["experiment_count"] == 4
    assert second.get("metrics") is None


# ---------------------------------------------------------------------------
# PostgreSQL recording and candidate state machine
# ---------------------------------------------------------------------------


def _external_candidate(
    store: ResearchStore, tmp_path: Path, values: pd.Series, *, name: str, run_kind: str
) -> dict:
    run = store.create_run(
        kind=run_kind,
        objective=f"Register external NLP factor {name} for evaluation.",
        dataset="snapshot-external",
        requested_by="researcher",
        budget={"loop_n": 0},
        config={},
        artifact_path=tmp_path,
    )
    code_path = tmp_path / f"{name}_factor.py"
    code_path.write_text('"""external provenance code"""\n', encoding="utf-8")
    values_path = tmp_path / f"{name}.parquet"
    values.rename(name).reset_index().to_parquet(values_path, index=False)
    return store.add_candidate(
        run["id"],
        name=name,
        description=f"Externally produced NLP factor {name}.",
        formulation="external NLP artifact",
        variables={"source": "external-test"},
        source_iteration=None,
        code_path=str(code_path),
        values_path=str(values_path),
        code_sha256=None,
        rdagent_decision=None,
        rdagent_feedback="external test candidate",
    )


def _evidence(store: ResearchStore, candidate_id: str, shape: str) -> dict:
    candidate = store.get_candidate(candidate_id)
    return ext.build_external_evidence(
        evaluation_shape=shape,
        config=ext.ExternalEvaluationConfig(),
        policy=ext.POLICY_BY_SHAPE[shape](),
        candidate=candidate,
        dataset_identity_sha256=DATASET_IDENTITY,
        periods=PERIODS,
        label_horizon_days=int(candidate["label_horizon_days"]),
        input_data_sha256=INPUT_DATA_SHA256,
    )


def test_external_event_evaluation_records_and_advances_candidate(
    tmp_path: Path, database_url: str
) -> None:
    store = ResearchStore(database_url)
    factor, label = _sparse_event_inputs(event_days=40)
    candidate = _external_candidate(
        store, tmp_path, factor, name="announcement_tone", run_kind="external_event_eval"
    )
    outcome = ext.evaluate_sparse_event_factor(
        factor,
        label,
        valid_start=PERIODS["valid_start"],
        valid_end=PERIODS["valid_end"],
        test_start=PERIODS["test_start"],
        test_end=PERIODS["test_end"],
    )
    entry = _event_entry(candidate["id"], outcome["metrics"])
    artifact = _write_result_artifact(
        tmp_path / "result.json", candidate["id"], entry["metrics"]
    )
    evaluation = store.record_external_evaluation(
        candidate["id"],
        policy=ExternalEventGatePolicy(),
        dataset="snapshot-external",
        dataset_identity_sha256=DATASET_IDENTITY,
        **PERIODS,
        evaluation_shape=ext.SHAPE_SPARSE_EVENT,
        metrics=entry["metrics"],
        insufficient_reasons=None,
        external_evidence=_evidence(store, candidate["id"], ext.SHAPE_SPARSE_EVENT),
        artifact_path=str(artifact),
    )
    assert evaluation["gate_status"] == "passed"
    assert evaluation["evaluator_version"] == "external-event-gate-v1"
    assert evaluation["statistical_contract_version"] == "research-statistics-v1-hac-bh-dsr"
    assert len(evaluation["evidence_sha256"]) == 64
    assert evaluation["recomputed_values_sha256"] == candidate["values_sha256"]
    assert evaluation["submitted_values_sha256"] == candidate["values_sha256"]
    assert evaluation["recompute_evidence"]["executor_version"] == EXTERNAL_EVALUATOR_VERSION
    assert evaluation["recompute_evidence"]["evaluation_shape"] == ext.SHAPE_SPARSE_EVENT
    updated = store.get_candidate(candidate["id"])
    assert updated["status"] == "gate_passed"
    assert updated["values_sha256"] == candidate["values_sha256"]
    assert updated["latest_evaluation"]["id"] == evaluation["id"]
    events = store.list_events(candidate["research_run_id"])
    assert "candidate.gate_passed" in {event["event_type"] for event in events}


def test_external_market_evaluation_records_with_market_policy(
    tmp_path: Path, database_url: str
) -> None:
    store = ResearchStore(database_url)
    factor, label = _market_inputs()
    candidate = _external_candidate(
        store, tmp_path, factor, name="news_sentiment_daily", run_kind="external_market_eval"
    )
    outcome = ext.evaluate_market_timeseries_factor(
        factor,
        label,
        valid_start=PERIODS["valid_start"],
        valid_end=PERIODS["valid_end"],
        test_start=PERIODS["test_start"],
        test_end=PERIODS["test_end"],
    )
    entry = _event_entry(candidate["id"], outcome["metrics"])
    artifact = _write_result_artifact(
        tmp_path / "result.json", candidate["id"], entry["metrics"]
    )
    evaluation = store.record_external_evaluation(
        candidate["id"],
        policy=MarketTimeseriesGatePolicy(),
        dataset="snapshot-external",
        dataset_identity_sha256=DATASET_IDENTITY,
        **PERIODS,
        evaluation_shape=ext.SHAPE_MARKET_TIMESERIES,
        metrics=entry["metrics"],
        insufficient_reasons=None,
        external_evidence=_evidence(store, candidate["id"], ext.SHAPE_MARKET_TIMESERIES),
        artifact_path=str(artifact),
    )
    assert evaluation["gate_status"] == "passed"
    assert evaluation["evaluator_version"] == "external-market-gate-v1"
    assert evaluation["metrics"]["benchmark_instrument"] == BENCHMARK
    assert store.get_candidate(candidate["id"])["status"] == "gate_passed"


def test_external_evaluation_insufficient_evidence_state_and_versioning(
    tmp_path: Path, database_url: str
) -> None:
    store = ResearchStore(database_url)
    factor, label = _sparse_event_inputs(event_days=8)
    candidate = _external_candidate(
        store, tmp_path, factor, name="irm_qa_sentiment_daily", run_kind="external_short_eval"
    )
    outcome = ext.evaluate_sparse_event_factor(
        factor,
        label,
        valid_start=PERIODS["valid_start"],
        valid_end=PERIODS["valid_end"],
        test_start=PERIODS["test_start"],
        test_end=PERIODS["test_end"],
    )
    assert outcome["status"] == "insufficient_evidence"
    evaluation = store.record_external_evaluation(
        candidate["id"],
        policy=ExternalEventGatePolicy(),
        dataset="snapshot-external",
        dataset_identity_sha256=DATASET_IDENTITY,
        **PERIODS,
        evaluation_shape=ext.SHAPE_SPARSE_EVENT,
        metrics=None,
        insufficient_reasons=outcome["reasons"],
        external_evidence=_evidence(store, candidate["id"], ext.SHAPE_SPARSE_EVENT),
        artifact_path=None,
    )
    assert evaluation["gate_status"] == "insufficient_evidence"
    assert evaluation["metrics"] == {}
    assert any("event days=8" in reason for reason in evaluation["gate_reasons"])
    assert store.get_candidate(candidate["id"])["status"] == "insufficient_evidence"
    events = store.list_events(candidate["research_run_id"])
    assert "candidate.insufficient_evidence" in {event["event_type"] for event in events}
    with pytest.raises(ValueError, match="must pass"):
        store.promote(candidate["id"], actor="portfolio-owner", reason="not enough evidence yet")

    # A new artifact version (more independent events) is a new candidate row
    # that can reach gate_passed; the insufficient record stays on the old one.
    richer, richer_label = _sparse_event_inputs(event_days=40, seed=12)
    versioned = _external_candidate(
        store, tmp_path, richer, name="irm_qa_sentiment_daily_v2", run_kind="external_v2_eval"
    )
    outcome = ext.evaluate_sparse_event_factor(
        richer,
        richer_label,
        valid_start=PERIODS["valid_start"],
        valid_end=PERIODS["valid_end"],
        test_start=PERIODS["test_start"],
        test_end=PERIODS["test_end"],
    )
    entry = _event_entry(versioned["id"], outcome["metrics"])
    artifact = _write_result_artifact(
        tmp_path / "result-v2.json", versioned["id"], entry["metrics"]
    )
    passed = store.record_external_evaluation(
        versioned["id"],
        policy=ExternalEventGatePolicy(),
        dataset="snapshot-external",
        dataset_identity_sha256=DATASET_IDENTITY,
        **PERIODS,
        evaluation_shape=ext.SHAPE_SPARSE_EVENT,
        metrics=entry["metrics"],
        insufficient_reasons=None,
        external_evidence=_evidence(store, versioned["id"], ext.SHAPE_SPARSE_EVENT),
        artifact_path=str(artifact),
    )
    assert passed["gate_status"] == "passed"
    assert store.get_candidate(versioned["id"])["status"] == "gate_passed"
    assert store.get_candidate(candidate["id"])["status"] == "insufficient_evidence"


def test_external_evaluation_rejects_tampered_values_artifact(
    tmp_path: Path, database_url: str
) -> None:
    store = ResearchStore(database_url)
    factor, label = _sparse_event_inputs(event_days=40)
    candidate = _external_candidate(
        store, tmp_path, factor, name="announcement_tone", run_kind="external_tamper_eval"
    )
    other, _ = _sparse_event_inputs(event_days=40, seed=99)
    other.rename("announcement_tone").reset_index().to_parquet(
        Path(candidate["values_path"]), index=False
    )
    outcome = ext.evaluate_sparse_event_factor(
        factor,
        label,
        valid_start=PERIODS["valid_start"],
        valid_end=PERIODS["valid_end"],
        test_start=PERIODS["test_start"],
        test_end=PERIODS["test_end"],
    )
    with pytest.raises(ValueError, match="factor values artifact changed"):
        store.record_external_evaluation(
            candidate["id"],
            policy=ExternalEventGatePolicy(),
            dataset="snapshot-external",
            dataset_identity_sha256=DATASET_IDENTITY,
            **PERIODS,
            evaluation_shape=ext.SHAPE_SPARSE_EVENT,
            metrics=outcome["metrics"],
            insufficient_reasons=None,
            external_evidence=_evidence(store, candidate["id"], ext.SHAPE_SPARSE_EVENT),
            artifact_path=None,
        )


def test_external_evaluation_rejects_unbound_evidence_and_wrong_policy(
    tmp_path: Path, database_url: str
) -> None:
    store = ResearchStore(database_url)
    factor, label = _sparse_event_inputs(event_days=40)
    candidate = _external_candidate(
        store, tmp_path, factor, name="announcement_tone", run_kind="external_binding_eval"
    )
    outcome = ext.evaluate_sparse_event_factor(
        factor,
        label,
        valid_start=PERIODS["valid_start"],
        valid_end=PERIODS["valid_end"],
        test_start=PERIODS["test_start"],
        test_end=PERIODS["test_end"],
    )
    entry = _event_entry(candidate["id"], outcome["metrics"])
    artifact = _write_result_artifact(
        tmp_path / "result.json", candidate["id"], entry["metrics"]
    )
    with pytest.raises(ValueError, match="requires ExternalEventGatePolicy"):
        store.record_external_evaluation(
            candidate["id"],
            policy=MarketTimeseriesGatePolicy(),
            dataset="snapshot-external",
            dataset_identity_sha256=DATASET_IDENTITY,
            **PERIODS,
            evaluation_shape=ext.SHAPE_SPARSE_EVENT,
            metrics=entry["metrics"],
            insufficient_reasons=None,
            external_evidence=_evidence(store, candidate["id"], ext.SHAPE_SPARSE_EVENT),
            artifact_path=str(artifact),
        )
    evidence = _evidence(store, candidate["id"], ext.SHAPE_SPARSE_EVENT)
    evidence["dataset_identity_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="not bound to the Qlib dataset"):
        store.record_external_evaluation(
            candidate["id"],
            policy=ExternalEventGatePolicy(),
            dataset="snapshot-external",
            dataset_identity_sha256=DATASET_IDENTITY,
            **PERIODS,
            evaluation_shape=ext.SHAPE_SPARSE_EVENT,
            metrics=entry["metrics"],
            insufficient_reasons=None,
            external_evidence=evidence,
            artifact_path=str(artifact),
        )
    mismatched = _write_result_artifact(
        tmp_path / "mismatched.json", candidate["id"], {**entry["metrics"], "ic": 0.5}
    )
    with pytest.raises(ValueError, match="do not match imported metrics"):
        store.record_external_evaluation(
            candidate["id"],
            policy=ExternalEventGatePolicy(),
            dataset="snapshot-external",
            dataset_identity_sha256=DATASET_IDENTITY,
            **PERIODS,
            evaluation_shape=ext.SHAPE_SPARSE_EVENT,
            metrics=entry["metrics"],
            insufficient_reasons=None,
            external_evidence=_evidence(store, candidate["id"], ext.SHAPE_SPARSE_EVENT),
            artifact_path=str(mismatched),
        )


def test_import_external_evaluations_end_to_end(tmp_path: Path, database_url: str) -> None:
    store = ResearchStore(database_url)
    factor, label = _sparse_event_inputs(event_days=40)
    candidate = _external_candidate(
        store, tmp_path, factor, name="announcement_tone", run_kind="external_import_eval"
    )
    outcome = ext.evaluate_sparse_event_factor(
        factor,
        label,
        valid_start=PERIODS["valid_start"],
        valid_end=PERIODS["valid_end"],
        test_start=PERIODS["test_start"],
        test_end=PERIODS["test_end"],
    )
    entry = {
        "candidate_id": candidate["id"],
        "status": outcome["status"],
        "shape": outcome["shape"],
        "metrics": outcome["metrics"],
        "reasons": outcome["reasons"],
        "experiment_family_id": "fam",
        "experiment_count": 1,
        "evidence": _evidence(store, candidate["id"], ext.SHAPE_SPARSE_EVENT),
    }
    failed = {"candidate_id": "unrelated", "status": "failed", "error": "boom"}
    result = {"status": "ok", "evaluations": [entry, failed]}
    ext.apply_family_bh_correction(result["evaluations"])
    artifact = _write_result_artifact(tmp_path / "result.json", candidate["id"], entry["metrics"])
    imported = ext.import_external_evaluations(
        store,
        result,
        dataset="snapshot-external",
        dataset_identity_sha256=DATASET_IDENTITY,
        periods=PERIODS,
        artifact_path=artifact,
    )
    assert len(imported) == 1
    assert imported[0]["gate_status"] == "passed"
    assert store.get_candidate(candidate["id"])["status"] == "gate_passed"
