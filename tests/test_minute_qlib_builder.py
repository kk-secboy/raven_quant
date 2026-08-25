import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from quant_data.execution_contract import (
    MINUTE_CANONICALIZATION_POLICY_VERSION,
    MINUTE_EXECUTION_CONTRACT_VERSION,
    MINUTE_SOURCE_UNIT_CONTRACTS,
)
from quant_data.minute_qlib_builder import (
    MINUTE_QLIB_DUCKDB_MEMORY_LIMIT,
    MINUTE_QLIB_DUCKDB_THREADS,
    MINUTE_QLIB_FIELDS,
    MinuteQlibBuilder,
)
from quant_data.qlib_minute_resample import (
    QLIB_MINUTE_RESAMPLE_CONTRACT_VERSION,
    resample_minute_frame,
)
from quant_data.snapshot_lineage import make_lineage_id

pytestmark = pytest.mark.no_database


def _snapshot_file_entry(snapshot: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(snapshot).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _refresh_snapshot_file_manifests(snapshot: Path) -> None:
    manifest_path = snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for dataset, entry in manifest["datasets"].items():
        files = sorted((snapshot / "parquet" / dataset).rglob("*.parquet"))
        entry["files"] = [_snapshot_file_entry(snapshot, path) for path in files]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _snapshot(
    tmp_path: Path,
    *,
    frequency: str = "1min",
    source_lineage_id: str | None = None,
) -> Path:
    source_lineage_id = source_lineage_id or "d" * 64
    snapshot = tmp_path / "minute-snapshot"
    target = snapshot / "parquet" / "etf_1m" / "partition_year=2024"
    target.mkdir(parents=True)
    first_time = "09:30:00" if frequency == "5min" else "09:31:00"
    second_time = "09:35:00" if frequency == "5min" else "09:32:00"
    pd.DataFrame(
        [
            {
                "ts_code": "510300.SH",
                "trade_time": f"2024-01-02 {first_time}",
                "open": 3.50,
                "high": 3.52,
                "low": 3.49,
                "close": 3.51,
                "vol": 1000,
                "amount": 3510,
            },
            {
                "ts_code": "510300.SH",
                "trade_time": f"2024-01-02 {second_time}",
                "open": 3.51,
                "high": 3.54,
                "low": 3.50,
                "close": 3.53,
                "vol": 1200,
                "amount": 4236,
            },
        ]
    ).to_parquet(target / "bars.parquet")
    bars_path = target / "bars.parquet"
    limits = snapshot / "parquet" / "stk_limit" / "partition_year=2024"
    limits.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ts_code": "510300.SH",
                "trade_date": "2024-01-02",
                "up_limit": 3.85,
                "down_limit": 3.15,
            }
        ]
    ).to_parquet(limits / "limits.parquet")
    limits_path = limits / "limits.parquet"
    lineage_configuration = {
        "start_date": "2024-01-02",
        "frequency": frequency,
        "provider": "test-provider",
    }
    lineage_id = make_lineage_id("minute-test", lineage_configuration)
    source_lineage_evidence = {
        "qlib_dataset": "daily-source",
        "qlib_dataset_identity_sha256": "1" * 64,
        "qlib_dataset_lineage_id": "2" * 64,
        "source_snapshot": "daily-snapshot",
        "source_snapshot_manifest_sha256": "3" * 64,
        "source_lineage_id": source_lineage_id,
        "calendar_start": "2024-01-02",
        "calendar_end": "2024-01-02",
    }
    source_lineage_evidence["evidence_sha256"] = hashlib.sha256(
        json.dumps(
            source_lineage_evidence,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    (snapshot / "manifest.json").write_text(
        json.dumps(
            {
                "frequency": frequency,
                "lineage_id": lineage_id,
                "lineage_contract": {
                    "kind": "minute-test",
                    "configuration": lineage_configuration,
                },
                "lineage_generation": 0,
                "parent_snapshot": None,
                "parent_manifest_sha256": None,
                "source_lineage_id": source_lineage_id,
                "source_lineage_evidence": source_lineage_evidence,
                "start_date": "2024-01-02",
                "end_date": "2024-01-02",
                "datasets": {
                    "etf_1m": {
                        "source_sha256": "b" * 64,
                        "files": [_snapshot_file_entry(snapshot, bars_path)],
                    },
                    "stk_limit": {
                        "source_sha256": "c" * 64,
                        "files": [_snapshot_file_entry(snapshot, limits_path)],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return snapshot


def test_builds_minute_qlib_staging_from_execution_snapshot(tmp_path: Path) -> None:
    builder = MinuteQlibBuilder(_snapshot(tmp_path))

    by_symbol = builder.build_staging(tmp_path / "minute-staging")

    frame = pd.read_parquet(by_symbol / "SH510300.parquet")
    assert frame["symbol"].tolist() == ["SH510300", "SH510300"]
    assert frame["date"].dt.strftime("%H:%M").tolist() == ["09:31", "09:32"]
    assert frame["vwap"].tolist() == pytest.approx([3.51, 3.53])
    assert frame["up_limit"].tolist() == pytest.approx([3.85, 3.85])
    assert frame["down_limit"].tolist() == pytest.approx([3.15, 3.15])
    assert pd.isna(frame["change"].iloc[0])
    assert frame["change"].iloc[1] == pytest.approx(3.53 / 3.51 - 1)
    assert set(MINUTE_QLIB_FIELDS).issubset(frame.columns)


def test_build_staging_bounds_duckdb_resources_to_staging_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_connect = duckdb.connect
    executed: list[str] = []

    class RecordingConnection:
        def __init__(self) -> None:
            self.connection = real_connect()

        def execute(self, query: str):
            executed.append(query)
            return self.connection.execute(query)

        def close(self) -> None:
            self.connection.close()

    monkeypatch.setattr(
        "quant_data.minute_qlib_builder.duckdb.connect",
        lambda: RecordingConnection(),
    )
    staging = tmp_path / "minute-staging"

    MinuteQlibBuilder(_snapshot(tmp_path)).build_staging(staging)

    temporary = staging.with_name(f".{staging.name}.tmp")
    spill = (temporary / ".duckdb-spill").resolve()
    assert f"SET memory_limit='{MINUTE_QLIB_DUCKDB_MEMORY_LIMIT}'" in executed
    assert f"SET threads={MINUTE_QLIB_DUCKDB_THREADS}" in executed
    assert f"SET temp_directory='{spill.as_posix()}'" in [
        statement.replace("\\\\", "/") for statement in executed
    ]
    assert not spill.exists()


def test_build_staging_closes_duckdb_and_cleans_spill_after_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_connect = duckdb.connect
    closed = False

    class FailingConnection:
        def __init__(self) -> None:
            self.connection = real_connect()

        def execute(self, query: str):
            if query.startswith("SET "):
                return self.connection.execute(query)
            raise RuntimeError("forced normalization failure")

        def close(self) -> None:
            nonlocal closed
            closed = True
            self.connection.close()

    monkeypatch.setattr(
        "quant_data.minute_qlib_builder.duckdb.connect",
        lambda: FailingConnection(),
    )
    staging = tmp_path / "failed-minute-staging"
    spill = staging.with_name(f".{staging.name}.tmp") / ".duckdb-spill"

    with pytest.raises(RuntimeError, match="forced normalization failure"):
        MinuteQlibBuilder(_snapshot(tmp_path)).build_staging(staging)

    assert closed is True
    assert not spill.exists()


def test_build_normalizes_hundredfold_provider_volume(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    source = next((snapshot / "parquet" / "etf_1m").rglob("*.parquet"))
    frame = pd.read_parquet(source)
    frame.loc[0, "vol"] = float(frame.loc[0, "vol"]) * 100.0
    frame.to_parquet(source, index=False)
    _refresh_snapshot_file_manifests(snapshot)

    by_symbol = MinuteQlibBuilder(snapshot).build_staging(tmp_path / "normalized-volume-staging")
    result = pd.read_parquet(by_symbol / "SH510300.parquet")

    assert result["volume"].iloc[0] == 1000.0
    assert result["vwap"].iloc[0] == pytest.approx(3.51)


def test_builds_staging_from_minute_datasets_with_different_source_columns(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)
    target = snapshot / "parquet" / "futures_1m" / "partition_year=2024"
    target.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ts_code": "IF2401.CFX",
                "trade_time": "2024-01-02 09:31:00",
                "open": 3300.0,
                "high": 3301.0,
                "low": 3299.0,
                "close": 3300.5,
                "vol": 120,
                "amount": 396060.0,
                "oi": 80000,
                "exchange_specific": "ignored",
            }
        ]
    ).to_parquet(target / "bars.parquet")
    manifest_path = snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["datasets"]["futures_1m"] = {"source_sha256": "e" * 64}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _refresh_snapshot_file_manifests(snapshot)

    by_symbol = MinuteQlibBuilder(snapshot).build_staging(tmp_path / "mixed-staging")

    future = pd.read_parquet(by_symbol / "CFXIF2401.parquet")
    assert future["oi"].tolist() == [80000.0]
    assert future["vwap"].tolist() == [3300.5]
    assert (by_symbol / "SH510300.parquet").is_file()


def test_zero_volume_minute_is_marked_paused(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    source = next((snapshot / "parquet" / "etf_1m").rglob("*.parquet"))
    frame = pd.read_parquet(source)
    frame.loc[len(frame)] = {
        "ts_code": "510300.SH",
        "trade_time": "2024-01-02 09:33:00",
        "open": 3.53,
        "high": 3.53,
        "low": 3.53,
        "close": 3.53,
        "vol": 0,
        "amount": 0,
    }
    frame.to_parquet(source, index=False)
    _refresh_snapshot_file_manifests(snapshot)

    by_symbol = MinuteQlibBuilder(snapshot).build_staging(tmp_path / "paused-staging")
    result = pd.read_parquet(by_symbol / "SH510300.parquet")

    assert result["paused"].tolist() == [0.0, 0.0, 1.0]
    assert result["vwap"].iloc[-1] == pytest.approx(3.53)


def test_rejects_non_minute_snapshot(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="supported minute"):
        MinuteQlibBuilder(_snapshot(tmp_path, frequency="day"))


def test_rejects_minute_snapshot_without_daily_source_evidence(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    manifest_path = snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("source_lineage_evidence")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="daily-source binding"):
        MinuteQlibBuilder(snapshot)


def _minute_bars(start: str, count: int, *, base: float, volume: float) -> list[dict]:
    hour, minute = (int(part) for part in start.split(":"))
    bars = []
    for index in range(count):
        open_price = base + index * 0.01
        bars.append(
            {
                "date": f"2024-01-02 {hour:02d}:{minute + index:02d}:00",
                "symbol": "SH510300",
                "open": open_price,
                "high": open_price + 0.02,
                "low": open_price - 0.02,
                "close": open_price + 0.01,
                "vwap": open_price + 0.01,
                "volume": volume,
                "factor": 1.0,
                "change": None,
                "amount": (open_price + 0.01) * volume,
                "paused": 0.0,
                "up_limit": 9.99,
                "down_limit": 0.01,
                "oi": None,
            }
        )
    return bars


def test_qllib_resamples_native_bars_with_ohlcv_semantics() -> None:
    frame = pd.DataFrame(
        [
            *_minute_bars("09:30", 15, base=3.50, volume=1000),
            *_minute_bars("09:45", 15, base=3.70, volume=800),
            # unclosed intraday tail: only 3 of the 15 source bars exist
            *_minute_bars("10:00", 3, base=3.90, volume=500),
        ]
    )

    def qlib_calendar(_index: pd.DatetimeIndex, _source: str, _target: str) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(
            ["2024-01-02 09:30:00", "2024-01-02 09:45:00", "2024-01-02 10:00:00"]
        )

    result = resample_minute_frame(
        frame,
        source_frequency="1min",
        target_frequency="15min",
        calendar_resampler=qlib_calendar,
    )

    assert result["date"].dt.strftime("%H:%M").tolist() == ["09:30", "09:45"]
    assert result["open"].tolist() == pytest.approx([3.50, 3.70])
    assert result["high"].tolist() == pytest.approx([3.66, 3.86])
    assert result["low"].tolist() == pytest.approx([3.48, 3.68])
    assert result["close"].tolist() == pytest.approx([3.65, 3.85])
    assert result["volume"].tolist() == pytest.approx([15_000, 12_000])
    assert result["vwap"].iloc[0] == pytest.approx(3.58)
    expected_amount = sum((3.70 + index * 0.01 + 0.01) * 800 for index in range(15))
    assert result["amount"].iloc[1] == pytest.approx(expected_amount)
    assert pd.isna(result["change"].iloc[0])
    assert result["change"].iloc[1] == pytest.approx(3.85 / 3.65 - 1)


def test_resample_drops_windows_without_full_source_coverage() -> None:
    frame = pd.DataFrame(_minute_bars("09:30", 14, base=3.50, volume=1000))

    def qlib_calendar(_index: pd.DatetimeIndex, _source: str, _target: str) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(["2024-01-02 09:30:00"])

    with pytest.raises(ValueError, match="no complete bars"):
        resample_minute_frame(
            frame,
            source_frequency="1min",
            target_frequency="15min",
            calendar_resampler=qlib_calendar,
        )


def test_resampled_builder_uses_pinned_qlib_runtime_and_records_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builder = MinuteQlibBuilder(_snapshot(tmp_path), target_frequency="30min")
    builder.build_staging(tmp_path / "native-canonical-audit")
    native = tmp_path / "native"
    native.mkdir()
    pd.DataFrame({"date": ["2024-01-02"], "symbol": ["SH510300"]}).to_parquet(
        native / "SH510300.parquet", index=False
    )
    staging = tmp_path / "resampled"
    captured: list[str] = []

    def fake_run(command: list[str], *, check: bool) -> None:
        assert check is True
        captured.extend(command)
        output = Path(command[command.index("--output") + 1])
        output.mkdir(parents=True)
        pd.DataFrame(
            {"date": [pd.Timestamp("2024-01-02 09:30")], "symbol": ["SH510300"]}
        ).to_parquet(output / "SH510300.parquet", index=False)

    monkeypatch.setattr("quant_data.minute_qlib_builder.subprocess.run", fake_run)
    by_symbol = builder.resample_staging(
        native_by_symbol=native,
        staging_path=staging,
        qlib_python="python",
        wsl_distro="Ubuntu",
    )

    assert (by_symbol / "SH510300.parquet").is_file()
    assert "resample_minute_qlib.py" in " ".join(captured)
    assert captured[captured.index("--source-frequency") + 1] == "1min"
    assert captured[captured.index("--target-frequency") + 1] == "30min"

    qlib_dir = tmp_path / "qlib-output"
    monkeypatch.setattr(
        "quant_data.minute_qlib_builder.QlibBuilder._snapshot_manifest_digest",
        lambda _self: "c" * 64,
    )
    builder._write_provenance(qlib_dir)
    provenance = json.loads((qlib_dir / "metadata" / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["frequency"] == "30min"
    assert provenance["source_frequency"] == "1min"
    assert provenance["resampled"] is True
    assert provenance["resample_contract_version"] == QLIB_MINUTE_RESAMPLE_CONTRACT_VERSION
    assert provenance["resample_engine"] == "qlib.utils.resam.resam_calendar"


def test_five_minute_snapshot_uses_native_qlib_frequency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builder = MinuteQlibBuilder(
        _snapshot(
            tmp_path,
            frequency="5min",
            source_lineage_id="d" * 64,
        )
    )
    staging_by_symbol = builder.build_staging(tmp_path / "staging")
    qlib_repo = tmp_path / "qlib"
    script = qlib_repo / "scripts" / "dump_bin.py"
    script.parent.mkdir(parents=True)
    script.write_text("# fixture", encoding="utf-8")
    qlib_dir = tmp_path / "qlib-output"
    captured: list[str] = []

    def fake_run(command: list[str], *, check: bool) -> None:
        assert check is True
        captured.extend(command)
        feature = qlib_dir / "features" / "sh510300"
        feature.mkdir(parents=True)
        (feature / "close.5min.bin").write_bytes(b"fixture")

    monkeypatch.setattr("quant_data.minute_qlib_builder.subprocess.run", fake_run)
    monkeypatch.setattr(
        "quant_data.minute_qlib_builder.QlibBuilder._snapshot_manifest_digest",
        lambda _self: "c" * 64,
    )
    builder.dump_bin(
        staging_by_symbol=staging_by_symbol,
        qlib_dir=qlib_dir,
        qlib_repo=qlib_repo,
        qlib_python="python",
        wsl_distro="Ubuntu",
    )

    assert captured[captured.index("--freq") + 1] == "5min"
    provenance = json.loads((qlib_dir / "metadata" / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["frequency"] == "5min"
    assert provenance["execution_contract_version"] == MINUTE_EXECUTION_CONTRACT_VERSION
    assert provenance["field_units"]["amount"] == "cny_yuan"
    assert provenance["field_units"]["vwap"] == (
        "source_price_cny_strict_amount_div_volume_else_close"
    )
    assert provenance["source_datasets"] == ["etf_1m"]
    assert provenance["source_unit_contracts"] == {"etf_1m": MINUTE_SOURCE_UNIT_CONTRACTS["etf_1m"]}
    assert len(provenance["dataset_lineage_id"]) == 64
    assert len(provenance["source_snapshot_lineage_id"]) == 64
    assert provenance["source_lineage_id"] == "d" * 64
    assert provenance["output_manifest"]["version"] == "qlib-output-files-v1"
    assert provenance["output_manifest"]["files"] == [
        {
            "path": "features/sh510300/close.5min.bin",
            "bytes": len(b"fixture"),
            "sha256": hashlib.sha256(b"fixture").hexdigest(),
        }
    ]


def test_fail_closes_stock_or_etf_amount_volume_unit_mismatch(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    source = next((snapshot / "parquet" / "etf_1m").rglob("*.parquet"))
    frame = pd.read_parquet(source)
    frame["vol"] *= 10
    frame.to_parquet(source, index=False)
    _refresh_snapshot_file_manifests(snapshot)

    builder = MinuteQlibBuilder(snapshot)
    by_symbol = builder.build_staging(tmp_path / "invalid-units")
    result = pd.read_parquet(by_symbol / "SH510300.parquet")

    assert result["volume"].tolist() == [0.0, 0.0]
    assert result["amount"].tolist() == [0.0, 0.0]
    assert result["paused"].tolist() == [1.0, 1.0]
    assert result["vwap"].tolist() == result["close"].tolist()
    assert builder.canonicalization_audit is not None
    assert builder.canonicalization_audit["nontradable_rows"] == 2


def test_five_minute_canonicalization_is_deterministic_and_financially_safe(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path, frequency="5min")
    source = next((snapshot / "parquet" / "etf_1m").rglob("*.parquet"))
    rows = [
        ("09:30", 2.00, 100, 200),
        ("09:31", 2.00, 100, 200),  # extra 1-minute row: exclude, never aggregate
        ("09:35", 2.00, 100, 2),  # provider hand-like volume: normalize to one share
        ("09:40", 10.00, 100, 1049),  # relative-only: retain, but VWAP falls back
        ("09:45", 0.45, 1, 1),  # whole-CNY amount rounding
        ("09:50", 0.15, 1000, 159),  # <= one price tick outside OHLC
        ("09:55", 1.00, 1, 0),  # positive volume with zero amount: non-tradable
        ("10:00", 1.00, 2, 0),  # second zero-amount anomaly
        ("10:05", 1.54, 64_632_064, 51_816_532),  # severe content anomaly
        ("15:30", 2.00, 100, 200),  # BSE block-trade confirmation session
    ]
    pd.DataFrame(
        [
            {
                "ts_code": "510300.SH",
                "trade_time": f"2024-01-02 {stamp}:00",
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "vol": volume,
                "amount": amount,
            }
            for stamp, price, volume, amount in rows
        ]
    ).to_parquet(source, index=False)
    _refresh_snapshot_file_manifests(snapshot)

    first = MinuteQlibBuilder(snapshot)
    first_by_symbol = first.build_staging(tmp_path / "canonical-first")
    second = MinuteQlibBuilder(snapshot)
    second_by_symbol = second.build_staging(tmp_path / "canonical-second")
    result = pd.read_parquet(first_by_symbol / "SH510300.parquet")
    repeat = pd.read_parquet(second_by_symbol / "SH510300.parquet")

    pd.testing.assert_frame_equal(result, repeat)
    assert result["date"].dt.strftime("%H:%M").tolist() == [
        "09:30",
        "09:35",
        "09:40",
        "09:45",
        "09:50",
        "09:55",
        "10:00",
        "10:05",
    ]
    assert result.loc[result["date"].dt.strftime("%H:%M") == "09:35", "volume"].item() == 1
    for stamp in ("09:40", "09:45", "09:50"):
        row = result.loc[result["date"].dt.strftime("%H:%M") == stamp].iloc[0]
        assert row["vwap"] == pytest.approx(row["close"])
        assert row["paused"] == 0.0
    severe = result.loc[
        result["date"].dt.strftime("%H:%M").isin(["09:55", "10:00", "10:05"])
    ]
    assert severe["volume"].tolist() == [0.0, 0.0, 0.0]
    assert severe["amount"].tolist() == [0.0, 0.0, 0.0]
    assert severe["paused"].tolist() == [1.0, 1.0, 1.0]
    assert severe["vwap"].tolist() == severe["close"].tolist()

    expected_counts = {
        "input_rows": 10,
        "output_rows": 8,
        "session_excluded_rows": 1,
        "offgrid_excluded_rows": 1,
        "offgrid_symbol_days": 1,
        "volume_normalized_rows": 1,
        "relative_fallback_rows": 1,
        "amount_rounding_fallback_rows": 1,
        "price_tick_fallback_rows": 1,
        "nontradable_rows": 3,
    }
    assert first.canonicalization_audit is not None
    assert second.canonicalization_audit is not None
    for key, value in expected_counts.items():
        assert first.canonicalization_audit[key] == value
    assert first.canonicalization_audit == second.canonicalization_audit
    assert first.canonicalization_audit["policy_version"] == (
        MINUTE_CANONICALIZATION_POLICY_VERSION
    )
    assert len(str(first.canonicalization_audit["event_sha256"])) == 64
    assert len(str(first.canonicalization_audit["summary_sha256"])) == 64


def _record_quality_gate(snapshot: Path, gate: dict) -> None:
    manifest_path = snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["quality_gate"] = gate
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_minute_builder_enforces_a_recorded_quality_gate(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    _record_quality_gate(snapshot, {"ok": False, "errors": ["daily: boom"]})

    with pytest.raises(ValueError, match="quality gate"):
        MinuteQlibBuilder(snapshot)


def test_minute_builder_accepts_a_passing_quality_gate(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    _record_quality_gate(
        snapshot,
        {"ok": True, "verified_at": "2026-07-17T00:00:00+00:00", "errors": []},
    )

    by_symbol = MinuteQlibBuilder(snapshot).build_staging(tmp_path / "staging")

    assert (by_symbol / "SH510300.parquet").is_file()


def test_minute_builder_validates_recorded_source_audit_contract(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    source_audit = {
        "dataset": "etf_1m",
        "policy": {"version": MINUTE_CANONICALIZATION_POLICY_VERSION},
        "source_rows": 2,
        "canonical_rows": 2,
        "audit_status": "pass_with_canonicalization",
    }
    source_audit["audit_sha256"] = hashlib.sha256(
        json.dumps(source_audit, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    _record_quality_gate(
        snapshot,
        {
            "ok": True,
            "errors": [],
            "minute_source_audits": {"etf_1m": source_audit},
        },
    )

    builder = MinuteQlibBuilder(snapshot)

    assert builder.source_minute_audit_sha256 == {
        "etf_1m": source_audit["audit_sha256"]
    }
    source_audit["source_rows"] = 3
    _record_quality_gate(
        snapshot,
        {
            "ok": True,
            "errors": [],
            "minute_source_audits": {"etf_1m": source_audit},
        },
    )
    with pytest.raises(ValueError, match="digest is inconsistent"):
        MinuteQlibBuilder(snapshot)
