from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from quant_platform.announcement_nlp import (
    LOGIC_FACTOR_NAME,
    PROMPT_VERSION,
    write_factor_artifact,
)
from quant_platform.event_market_response import (
    LABEL_ROLE,
    _validated_logic_source,
    build_event_market_response_labels,
    process_event_market_response,
    write_event_market_response_labels,
)

pytestmark = pytest.mark.no_database


def _benchmark() -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=7, freq="B")
    closes = [100.0, 101.0, 102.0, 101.0, 103.0, 104.0, 105.0]
    return pd.DataFrame(
        {
            "ts_code": "000300.SH",
            "trade_date": dates,
            "pre_close": [99.0, *closes[:-1]],
            "close": closes,
        }
    )


def _stock() -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=7, freq="B")
    closes = [10.0, 11.0, 12.0, 11.5, 12.5, 13.0, 13.2]
    return pd.DataFrame(
        {
            "ts_code": "000001.SZ",
            "trade_date": dates,
            "pre_close": [9.8, *closes[:-1]],
            "close": closes,
            "amount": [90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
        }
    )


def _fields(direction: str = "positive") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "process_key": ["sha:v2:model"],
            "ts_code": ["000001.SZ"],
            "available_at": [pd.Timestamp("2024-01-03")],
            "impact_direction": [direction],
        }
    )


def test_build_labels_uses_pre_close_and_explicit_observation_time() -> None:
    labels = build_event_market_response_labels(
        _fields(), _stock(), _benchmark(), horizons=(1, 3), trailing_sessions=1
    )

    row = labels.iloc[0]
    # 1d stock return is the first reaction session: 11 / 10 - 1.
    assert row["stock_return_1d"] == pytest.approx(0.10)
    assert row["benchmark_return_1d"] == pytest.approx(0.01)
    assert row["abnormal_return_1d"] == pytest.approx(0.09)
    assert row["market_recognition_1d"] == pytest.approx(0.09)
    assert row["outcome_end_1d"] == pd.Timestamp("2024-01-03")
    # A post-event outcome is not declared observable until the next session.
    assert row["label_available_at_1d"] == pd.Timestamp("2024-01-04")

    assert row["stock_return_3d"] == pytest.approx(0.15)
    assert row["benchmark_return_3d"] == pytest.approx(0.01)
    assert row["outcome_end_3d"] == pd.Timestamp("2024-01-05")
    assert row["label_available_at_3d"] == pd.Timestamp("2024-01-08")


def test_negative_direction_flips_recognition_and_neutral_stays_unknown() -> None:
    negative = build_event_market_response_labels(
        _fields("negative"), _stock(), _benchmark(), horizons=(1,)
    ).iloc[0]
    neutral = build_event_market_response_labels(
        _fields("neutral"), _stock(), _benchmark(), horizons=(1,)
    ).iloc[0]

    assert negative["market_recognition_1d"] == pytest.approx(
        -negative["abnormal_return_1d"]
    )
    assert pd.isna(neutral["market_recognition_1d"])


def test_incomplete_or_suspended_horizon_is_null_not_shifted() -> None:
    stock = _stock()
    # Remove the exact 3-session outcome date. The builder must not pick the
    # security's next row and silently create a later-date label.
    stock = stock[stock["trade_date"] != pd.Timestamp("2024-01-05")]
    labels = build_event_market_response_labels(
        _fields(), stock, _benchmark(), horizons=(3, 20)
    )

    row = labels.iloc[0]
    assert not bool(row["complete_3d"])
    assert pd.isna(row["abnormal_return_3d"])
    assert pd.isna(row["outcome_end_3d"])
    assert not bool(row["complete_20d"])
    assert pd.isna(row["label_available_at_20d"])


def test_indexed_lookup_matches_scalar_base_close_and_amount_oracle() -> None:
    dates = pd.date_range("2024-01-02", periods=12, freq="B")
    stock = pd.DataFrame(
        {
            "ts_code": "000001.SZ",
            "trade_date": dates,
            "pre_close": [9.0, *[float(value) for value in range(10, 21)]],
            "close": [float(value) for value in range(10, 22)],
            "amount": [10.0, 20.0, 0.0, None, 30.0, 40.0, 50.0, 60.0, 0.0, 80.0, 90.0, 100.0],
        }
    )
    # Exercise the exact scalar fallback used before indexing: when pre_close
    # is unusable, the previous security close is the reaction-session base.
    stock.loc[stock["trade_date"] == dates[7], "pre_close"] = None
    benchmark = pd.DataFrame(
        {
            "ts_code": "000300.SH",
            "trade_date": dates,
            "pre_close": [99.0, *[float(value) for value in range(100, 111)]],
            "close": [float(value) for value in range(100, 112)],
        }
    )
    fields = pd.DataFrame(
        {
            "process_key": ["later", "early"],
            "ts_code": "000001.SZ",
            "available_at": [dates[7], dates[6]],
            "impact_direction": "positive",
        }
    )

    labels = build_event_market_response_labels(
        fields,
        stock,
        benchmark,
        horizons=(1, 3),
        trailing_sessions=7,
    ).set_index("process_key")

    def scalar_base_close(frame: pd.DataFrame, position: int) -> float | None:
        pre_close = frame.iloc[position]["pre_close"]
        if pd.notna(pre_close) and float(pre_close) > 0:
            return float(pre_close)
        if position > 0:
            previous_close = frame.iloc[position - 1]["close"]
            if pd.notna(previous_close) and float(previous_close) > 0:
                return float(previous_close)
        return None

    def scalar_amount_surprise(start: int, end_date: pd.Timestamp) -> float | None:
        history = stock.iloc[max(0, start - 7) : start]["amount"]
        event = stock[
            (stock["trade_date"] >= stock.iloc[start]["trade_date"])
            & (stock["trade_date"] <= end_date)
        ]["amount"]
        history = history[(history > 0) & history.notna()]
        event = event[(event > 0) & event.notna()]
        if len(history) < 5 or event.empty:
            return None
        baseline = float(history.median())
        return None if baseline <= 0 else float(event.mean()) / baseline - 1.0

    for process_key, start in (("early", 6), ("later", 7)):
        stock_base = scalar_base_close(stock, start)
        benchmark_base = scalar_base_close(benchmark, start)
        for horizon in (1, 3):
            end = start + horizon - 1
            prefix = f"{horizon}d"
            expected_stock_return = float(stock.iloc[end]["close"]) / stock_base - 1.0
            expected_benchmark_return = float(benchmark.iloc[end]["close"]) / benchmark_base - 1.0
            row = labels.loc[process_key]
            assert row[f"stock_return_{prefix}"] == pytest.approx(expected_stock_return)
            assert row[f"benchmark_return_{prefix}"] == pytest.approx(expected_benchmark_return)
            expected_amount = scalar_amount_surprise(start, dates[end])
            if expected_amount is None:
                assert pd.isna(row[f"amount_surprise_{prefix}"])
            else:
                assert row[f"amount_surprise_{prefix}"] == pytest.approx(expected_amount)


def test_last_snapshot_session_is_not_published_without_next_session() -> None:
    fields = _fields()
    fields["available_at"] = pd.Timestamp("2024-01-10")
    row = build_event_market_response_labels(
        fields, _stock(), _benchmark(), horizons=(1,)
    ).iloc[0]

    assert not bool(row["complete_1d"])
    assert pd.isna(row["outcome_end_1d"])
    assert pd.isna(row["label_available_at_1d"])


def test_conflicting_daily_duplicates_fail_closed() -> None:
    stock = pd.concat(
        [
            _stock(),
            pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "trade_date": [pd.Timestamp("2024-01-03")],
                    "pre_close": [10.0],
                    "close": [99.0],
                    "amount": [100.0],
                }
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="conflicting duplicate"):
        build_event_market_response_labels(_fields(), stock, _benchmark(), horizons=(1,))


def test_manifest_marks_artifact_as_training_label_only(tmp_path: Path) -> None:
    labels = build_event_market_response_labels(
        _fields(), _stock(), _benchmark(), horizons=(1, 3)
    )
    summary = write_event_market_response_labels(
        labels,
        tmp_path,
        horizons=(1, 3),
        source={"snapshot": "fixture"},
        now=datetime(2026, 8, 8, tzinfo=UTC),
    )

    manifest = json.loads(summary.manifest_path.read_text(encoding="utf-8"))
    assert manifest["role"] == LABEL_ROLE
    assert "factor_candidates" in manifest["forbidden_consumers"]
    assert "qlib_inference_features" in manifest["forbidden_consumers"]
    assert manifest["complete_by_horizon"] == {"1d": 1, "3d": 1}
    assert summary.rows == 1
    assert pd.read_parquet(summary.labels_path)["label_role"].unique().tolist() == [
        LABEL_ROLE
    ]


def test_logic_source_resolves_checksum_bound_mixed_models(tmp_path: Path) -> None:
    artifact = tmp_path / f"{LOGIC_FACTOR_NAME}.parquet"
    artifact.write_bytes(b"governed-mixed-model-artifact")
    manifest_path = tmp_path / f"{LOGIC_FACTOR_NAME}.json"
    manifest_path.write_text(
        json.dumps(
            {
                "factor": LOGIC_FACTOR_NAME,
                "artifact": artifact.name,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "source": {
                    "dataset": "announcement_nlp_fields",
                    "prompt_version": PROMPT_VERSION,
                    "model": "mixed[model-a,model-b]",
                    "scope": {"models": ["model-b", "model-a"]},
                },
            }
        ),
        encoding="utf-8",
    )

    _, model, models = _validated_logic_source(
        manifest_path, prompt_version=PROMPT_VERSION
    )
    assert model == "mixed[model-a,model-b]"
    assert models == ("model-a", "model-b")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["source"]["model"] = "mixed[model-a,model-c]"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="incompatible source"):
        _validated_logic_source(manifest_path, prompt_version=PROMPT_VERSION)


def _mixed_publication_fields() -> pd.DataFrame:
    fields = pd.concat([_fields(), _fields(), _fields()], ignore_index=True)
    fields["process_key"] = ["a-flash", "a-pro", "b-pro"]
    fields["source_sha256"] = ["a" * 64, "a" * 64, "b" * 64]
    fields["model"] = ["flash", "pro", "pro"]
    fields["prompt_version"] = PROMPT_VERSION
    fields["processed_at"] = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-02"])
    fields["impact_horizon"] = "short_term"
    fields["confidence"] = 0.8
    return fields


def test_logic_source_verifies_publication_models_separately_from_processing_scope(
    tmp_path: Path,
) -> None:
    fields = _mixed_publication_fields()
    artifact = write_factor_artifact(
        fields, tmp_path, name=LOGIC_FACTOR_NAME, model="mixed[flash,pro]",
        now=datetime(2025, 1, 3, tzinfo=UTC), process_keys={"a-flash", "b-pro"},
        processing_process_keys={"a-flash"},
        source_scope={"requested_model": "flash", "models": ["flash"]},
    )
    manifest_path = artifact["manifest_path"]
    _, model, models = _validated_logic_source(
        manifest_path, prompt_version=PROMPT_VERSION, fields=fields
    )
    assert model == "mixed[flash,pro]" and models == ("flash", "pro")

    changed = fields.copy()
    changed.loc[2, "process_key"] = "changed-key"
    with pytest.raises(RuntimeError, match="scope checksum verification"):
        _validated_logic_source(manifest_path, prompt_version=PROMPT_VERSION, fields=changed)
    manifest = json.loads(manifest_path.read_text())
    manifest["source"]["model"] = "mixed[flash,unpublished]"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="incompatible source"):
        _validated_logic_source(manifest_path, prompt_version=PROMPT_VERSION, fields=fields)


@pytest.mark.parametrize("publication_scope", [False, True])
def test_process_requires_verified_snapshot_and_binds_source_hashes(
    tmp_path: Path, publication_scope: bool
) -> None:
    snapshot_name = "cn-fixture"
    snapshot = tmp_path / "snapshots" / snapshot_name
    snapshot.mkdir(parents=True)
    (snapshot / "verification.json").write_text(
        json.dumps({"ok": True, "errors": []}), encoding="utf-8"
    )
    fields = _fields()
    fields["prompt_version"] = PROMPT_VERSION
    fields["model"] = "test-model"
    if publication_scope:
        fields = _mixed_publication_fields()
    fields_dir = tmp_path / "announcements" / "nlp"
    fields_dir.mkdir(parents=True)
    fields.to_parquet(fields_dir / "fields.parquet", index=False)
    factors_dir = fields_dir / "factors"
    factors_dir.mkdir()
    logic_artifact_path = factors_dir / f"{LOGIC_FACTOR_NAME}.parquet"
    logic_artifact_path.write_bytes(b"governed-logic-fixture")
    (factors_dir / f"{LOGIC_FACTOR_NAME}.json").write_text(
        json.dumps(
            {
                "factor": LOGIC_FACTOR_NAME,
                "artifact": logic_artifact_path.name,
                "sha256": hashlib.sha256(logic_artifact_path.read_bytes()).hexdigest(),
                "source": {
                    "dataset": "announcement_nlp_fields",
                    "prompt_version": PROMPT_VERSION,
                    "model": "test-model",
                },
            }
        ),
        encoding="utf-8",
    )
    if publication_scope:
        write_factor_artifact(
            fields, factors_dir, name=LOGIC_FACTOR_NAME, model="mixed[flash,pro]",
            now=datetime(2025, 1, 3, tzinfo=UTC), process_keys={"a-flash", "b-pro"},
            processing_process_keys={"a-flash"},
            source_scope={"requested_model": "flash", "models": ["flash"]},
        )

    datasets = {}
    for dataset, frame in (("daily", _stock()), ("index_daily", _benchmark())):
        target = snapshot / "parquet" / dataset / "partition_year=2024" / "partition_month=1"
        target.mkdir(parents=True)
        parquet_path = target / "data.parquet"
        frame.to_parquet(parquet_path, index=False)
        datasets[dataset] = {"files": [{
            "path": parquet_path.relative_to(snapshot).as_posix(),
            "bytes": parquet_path.stat().st_size,
            "sha256": hashlib.sha256(parquet_path.read_bytes()).hexdigest(),
        }]}
    (snapshot / "manifest.json").write_text(
        json.dumps({"name": snapshot_name, "profile": "full", "end_date": "2024-01-10",
                    "datasets": datasets}), encoding="utf-8"
    )

    summary = process_event_market_response(
        tmp_path, snapshot_name=snapshot_name, horizons=(1, 3)
    )
    manifest = json.loads(summary.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"]["snapshot_name"] == snapshot_name
    assert len(manifest["source"]["snapshot_manifest_sha256"]) == 64
    assert manifest["source"]["prompt_version"] == PROMPT_VERSION
    assert manifest["source"]["model"] == (
        "mixed[flash,pro]" if publication_scope else "test-model"
    )
    assert manifest["source"]["models"] == (
        ["flash", "pro"] if publication_scope else ["test-model"]
    )
    assert len(manifest["source"]["logic_factor_manifest_sha256"]) == 64
    labels = pd.read_parquet(summary.labels_path)
    assert set(labels["process_key"]) == (
        {"a-flash", "b-pro"} if publication_scope else {"sha:v2:model"}
    )

    original_bars = parquet_path.read_bytes()
    parquet_path.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="market bars failed snapshot checksum"):
        process_event_market_response(tmp_path, snapshot_name=snapshot_name, horizons=(1,))
    parquet_path.write_bytes(original_bars)

    logic_artifact_path.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="checksum verification"):
        process_event_market_response(tmp_path, snapshot_name=snapshot_name, horizons=(1,))

    (snapshot / "verification.json").write_text(
        json.dumps({"ok": False, "errors": ["broken"]}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="quality gate"):
        process_event_market_response(tmp_path, snapshot_name=snapshot_name, horizons=(1,))
