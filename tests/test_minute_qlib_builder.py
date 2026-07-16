import json
from pathlib import Path

import pandas as pd
import pytest

from quant_data.execution_contract import (
    MINUTE_EXECUTION_CONTRACT_VERSION,
    MINUTE_SOURCE_UNIT_CONTRACTS,
)
from quant_data.minute_qlib_builder import MINUTE_QLIB_FIELDS, MinuteQlibBuilder

pytestmark = pytest.mark.no_database


def _snapshot(tmp_path: Path, *, frequency: str = "1min") -> Path:
    snapshot = tmp_path / "minute-snapshot"
    target = snapshot / "parquet" / "etf_1m" / "partition_year=2024"
    target.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ts_code": "510300.SH",
                "trade_time": "2024-01-02 09:31:00",
                "open": 3.50,
                "high": 3.52,
                "low": 3.49,
                "close": 3.51,
                "vol": 1000,
                "amount": 3510,
            },
            {
                "ts_code": "510300.SH",
                "trade_time": "2024-01-02 09:32:00",
                "open": 3.51,
                "high": 3.54,
                "low": 3.50,
                "close": 3.53,
                "vol": 1200,
                "amount": 4236,
            },
        ]
    ).to_parquet(target / "bars.parquet")
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
    (snapshot / "manifest.json").write_text(
        json.dumps(
            {
                "frequency": frequency,
                "lineage_id": "a" * 64,
                "start_date": "2024-01-02",
                "end_date": "2024-01-02",
                "datasets": {
                    "etf_1m": {"source_sha256": "b" * 64},
                    "stk_limit": {"source_sha256": "c" * 64},
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
    manifest["datasets"]["futures_1m"] = {}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

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

    by_symbol = MinuteQlibBuilder(snapshot).build_staging(tmp_path / "paused-staging")
    result = pd.read_parquet(by_symbol / "SH510300.parquet")

    assert result["paused"].tolist() == [0.0, 0.0, 1.0]
    assert result["vwap"].iloc[-1] == pytest.approx(3.53)


def test_rejects_non_minute_snapshot(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="supported minute"):
        MinuteQlibBuilder(_snapshot(tmp_path, frequency="day"))


def test_five_minute_snapshot_uses_native_qlib_frequency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builder = MinuteQlibBuilder(_snapshot(tmp_path, frequency="5min"))
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
        staging_by_symbol=tmp_path / "staging",
        qlib_dir=qlib_dir,
        qlib_repo=qlib_repo,
        qlib_python="python",
        wsl_distro="Ubuntu",
    )

    assert captured[captured.index("--freq") + 1] == "5min"
    provenance = json.loads(
        (qlib_dir / "metadata" / "provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["frequency"] == "5min"
    assert provenance["execution_contract_version"] == MINUTE_EXECUTION_CONTRACT_VERSION
    assert provenance["source_datasets"] == ["etf_1m"]
    assert provenance["source_unit_contracts"] == {
        "etf_1m": MINUTE_SOURCE_UNIT_CONTRACTS["etf_1m"]
    }


def test_rejects_stock_or_etf_amount_volume_unit_mismatch(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    source = next((snapshot / "parquet" / "etf_1m").rglob("*.parquet"))
    frame = pd.read_parquet(source)
    frame["vol"] *= 100
    frame.to_parquet(source, index=False)

    with pytest.raises(RuntimeError, match="share-volume/CNY-amount"):
        MinuteQlibBuilder(snapshot).build_staging(tmp_path / "invalid-units")
