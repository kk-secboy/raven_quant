from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from quant_platform.announcement_nlp import LOGIC_FACTOR_NAME, PROMPT_VERSION
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


def test_process_requires_verified_snapshot_and_binds_source_hashes(tmp_path: Path) -> None:
    snapshot_name = "cn-fixture"
    snapshot = tmp_path / "snapshots" / snapshot_name
    snapshot.mkdir(parents=True)
    (snapshot / "verification.json").write_text(
        json.dumps({"ok": True, "errors": []}), encoding="utf-8"
    )
    (snapshot / "manifest.json").write_text(
        json.dumps({"snapshot": snapshot_name}), encoding="utf-8"
    )
    fields = _fields()
    fields["prompt_version"] = PROMPT_VERSION
    fields["model"] = "test-model"
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

    for dataset, frame in (("daily", _stock()), ("index_daily", _benchmark())):
        target = snapshot / "parquet" / dataset / "partition_year=2024" / "partition_month=1"
        target.mkdir(parents=True)
        frame.to_parquet(target / "data.parquet", index=False)

    summary = process_event_market_response(
        tmp_path, snapshot_name=snapshot_name, horizons=(1, 3)
    )
    manifest = json.loads(summary.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"]["snapshot_name"] == snapshot_name
    assert len(manifest["source"]["snapshot_manifest_sha256"]) == 64
    assert manifest["source"]["prompt_version"] == PROMPT_VERSION
    assert manifest["source"]["model"] == "test-model"
    assert manifest["source"]["models"] == ["test-model"]
    assert len(manifest["source"]["logic_factor_manifest_sha256"]) == 64

    logic_artifact_path.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="checksum verification"):
        process_event_market_response(tmp_path, snapshot_name=snapshot_name, horizons=(1,))

    (snapshot / "verification.json").write_text(
        json.dumps({"ok": False, "errors": ["broken"]}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="quality gate"):
        process_event_market_response(tmp_path, snapshot_name=snapshot_name, horizons=(1,))
