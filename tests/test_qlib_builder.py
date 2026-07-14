import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from quant_data.qlib_builder import QlibBuilder, _to_wsl_path

pytestmark = pytest.mark.no_database


def _write_market_control_snapshot(
    tmp_path: Path,
    *,
    ts_code: str,
    up_limit: float,
    down_limit: float,
) -> Path:
    snapshot = tmp_path / "snapshot"
    values = {
        "daily": {
            "ts_code": ts_code,
            "trade_date": "2024-01-02",
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.0,
            "vol": 100.0,
            "amount": 1000.0,
            "pct_chg": 0.0,
        },
        "adj_factor": {
            "ts_code": ts_code,
            "trade_date": "2024-01-02",
            "adj_factor": 1.0,
        },
        "stk_limit": {
            "ts_code": ts_code,
            "trade_date": "2024-01-02",
            "up_limit": up_limit,
            "down_limit": down_limit,
        },
    }
    for dataset, row in values.items():
        target = snapshot / "parquet" / dataset / "partition_year=2024"
        target.mkdir(parents=True)
        pd.DataFrame([row]).to_parquet(target / "data.parquet")
    return snapshot


def test_builds_per_symbol_normalized_qlib_staging(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    daily_dir = snapshot / "parquet" / "daily" / "partition_year=2024" / "partition_month=1"
    adj_dir = snapshot / "parquet" / "adj_factor" / "partition_year=2024" / "partition_month=1"
    limit_dir = snapshot / "parquet" / "stk_limit" / "partition_year=2024" / "partition_month=1"
    daily_dir.mkdir(parents=True)
    adj_dir.mkdir(parents=True)
    limit_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": pd.Timestamp("2024-01-02").date(),
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.0,
                "vol": 100.0,
                "amount": 100.0,
                "pct_chg": 0.0,
            },
            {
                "ts_code": "000001.SZ",
                "trade_date": pd.Timestamp("2024-01-03").date(),
                "open": 5.0,
                "high": 5.5,
                "low": 4.5,
                "close": 5.0,
                "vol": 50.0,
                "amount": 25.0,
                "pct_chg": 0.0,
            },
        ]
    ).to_parquet(daily_dir / "data.parquet")
    pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": pd.Timestamp("2024-01-02").date(),
                "adj_factor": 1.0,
            },
            {
                "ts_code": "000001.SZ",
                "trade_date": pd.Timestamp("2024-01-03").date(),
                "adj_factor": 2.0,
            },
        ]
    ).to_parquet(adj_dir / "data.parquet")
    pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": pd.Timestamp(day).date(),
                "up_limit": up,
                "down_limit": down,
            }
            for day, up, down in (
                ("2024-01-02", 11.0, 9.0),
                ("2024-01-03", 5.5, 4.5),
            )
        ]
    ).to_parquet(limit_dir / "data.parquet")

    by_symbol = QlibBuilder(snapshot).build_staging(tmp_path / "staging")
    frame = pd.read_parquet(by_symbol / "SZ000001.parquet")
    assert frame["symbol"].tolist() == ["SZ000001", "SZ000001"]
    assert frame["close"].tolist() == pytest.approx([1.0, 1.0])
    assert frame["factor"].tolist() == pytest.approx([0.1, 0.2])
    assert frame["volume"].tolist() == pytest.approx([1000.0, 250.0])
    assert frame["vwap"].tolist() == pytest.approx([1.0, 1.0])
    assert frame["up_limit"].tolist() == pytest.approx([1.1, 1.1])
    assert frame["down_limit"].tolist() == pytest.approx([0.9, 0.9])


def test_accepts_tushare_unrestricted_price_limit_sentinel(tmp_path: Path) -> None:
    snapshot = _write_market_control_snapshot(
        tmp_path,
        ts_code="920690.BJ",
        up_limit=99999.99,
        down_limit=0.0,
    )

    by_symbol = QlibBuilder(snapshot).build_staging(tmp_path / "staging")
    frame = pd.read_parquet(by_symbol / "BJ920690.parquet")

    assert frame["up_limit"].iloc[0] == pytest.approx(9999.999)
    assert frame["down_limit"].iloc[0] == 0.0


def test_rejects_non_sentinel_zero_price_limit(tmp_path: Path) -> None:
    snapshot = _write_market_control_snapshot(
        tmp_path,
        ts_code="000001.SZ",
        up_limit=11.0,
        down_limit=0.0,
    )

    with pytest.raises(RuntimeError, match="price limits"):
        QlibBuilder(snapshot).build_staging(tmp_path / "staging")


def test_adds_normalized_index_staging(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    daily_dir = snapshot / "parquet" / "daily" / "partition_year=2024"
    adj_dir = snapshot / "parquet" / "adj_factor" / "partition_year=2024"
    index_dir = snapshot / "parquet" / "index_daily" / "partition_year=2024"
    limit_dir = snapshot / "parquet" / "stk_limit" / "partition_year=2024"
    daily_dir.mkdir(parents=True)
    adj_dir.mkdir(parents=True)
    index_dir.mkdir(parents=True)
    limit_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "2024-01-02",
                "open": 10,
                "high": 10,
                "low": 10,
                "close": 10,
                "vol": 10,
                "amount": 100,
                "pct_chg": 0,
            }
        ]
    ).to_parquet(daily_dir / "data.parquet")
    pd.DataFrame(
        [{"ts_code": "000001.SZ", "trade_date": "2024-01-02", "adj_factor": 1}]
    ).to_parquet(adj_dir / "data.parquet")
    pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "2024-01-02",
                "up_limit": 11.0,
                "down_limit": 9.0,
            }
        ]
    ).to_parquet(limit_dir / "data.parquet")
    pd.DataFrame(
        [
            {
                "ts_code": "000300.SH",
                "trade_date": "2024-01-02",
                "open": 3000,
                "high": 3030,
                "low": 2970,
                "close": 3000,
                "vol": 100,
                "amount": 30000,
                "pct_chg": 0,
            }
        ]
    ).to_parquet(index_dir / "data.parquet")

    by_symbol = QlibBuilder(snapshot).build_staging(tmp_path / "staging")
    index = pd.read_parquet(by_symbol / "SH000300.parquet")
    assert index["close"].tolist() == pytest.approx([1.0])
    assert index["vwap"].tolist() == pytest.approx([1.0])


def test_windows_path_maps_to_wsl() -> None:
    assert _to_wsl_path(Path("E:/projects/qlib")) == "/mnt/e/projects/qlib"


def test_writes_point_in_time_industry_metadata(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    source = snapshot / "parquet" / "index_member_all"
    source.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "l1_code": "801780.SI",
                "in_date": "20210101",
                "out_date": None,
            },
            {
                "ts_code": "600000.SH",
                "l1_code": "801780.SI",
                "in_date": "20220101",
                "out_date": "20241231",
            },
        ]
    ).to_parquet(source / "members.parquet")
    weights_source = snapshot / "parquet" / "index_weight"
    weights_source.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "index_code": "000300.SH",
                "con_code": "000001.SZ",
                "trade_date": "20240131",
                "weight": 4.5,
            }
        ]
    ).to_parquet(weights_source / "weights.parquet")
    style_source = snapshot / "parquet" / "daily_basic"
    style_source.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20240131",
                "total_mv": 125000.0,
            }
        ]
    ).to_parquet(style_source / "styles.parquet")
    qlib_dir = tmp_path / "qlib"

    QlibBuilder(snapshot)._write_portfolio_metadata(qlib_dir)

    metadata = pd.read_parquet(qlib_dir / "metadata" / "industry_memberships.parquet")
    assert metadata["instrument"].tolist() == ["SH600000", "SZ000001"]
    assert metadata["industry"].tolist() == ["801780.SI", "801780.SI"]
    assert metadata.loc[metadata["instrument"] == "SH600000", "out_date"].notna().all()
    weights = pd.read_parquet(qlib_dir / "metadata" / "benchmark_weights.parquet")
    assert weights.loc[0, "benchmark"] == "SH000300"
    assert weights.loc[0, "instrument"] == "SZ000001"
    assert weights.loc[0, "weight"] == pytest.approx(0.045)
    styles = pd.read_parquet(qlib_dir / "metadata" / "style_exposures.parquet")
    assert styles.loc[0, "instrument"] == "SZ000001"
    assert styles.loc[0, "log_market_cap"] == pytest.approx(11.736069, rel=1e-6)


def test_writes_reproducible_qlib_dataset_provenance(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    manifest = {
        "name": "snapshot",
        "profile": "full",
        "datasets": {"daily": {"rows": 0, "source_sha256": "a" * 64, "files": []}},
    }
    manifest_path = snapshot / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    qlib_dir = tmp_path / "qlib"

    QlibBuilder(snapshot)._write_provenance(qlib_dir)

    provenance = json.loads(
        (qlib_dir / "metadata" / "provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["snapshot_name"] == "snapshot"
    assert provenance["snapshot_manifest_sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    assert len(provenance["dataset_identity_sha256"]) == 64
    assert len(provenance["qlib_builder_sha256"]) == 64
    assert provenance["lineage_verified"] is False
    assert provenance["dataset_lineage_id"] is None


def test_derives_stable_qlib_lineage_only_from_verified_source_lineage(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    manifest = {
        "name": "snapshot",
        "profile": "full",
        "lineage_id": "a" * 64,
        "lineage_generation": 2,
        "datasets": {"daily": {"rows": 0, "source_sha256": "b" * 64, "files": []}},
    }
    (snapshot / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    qlib_dir = tmp_path / "qlib"

    QlibBuilder(snapshot)._write_provenance(qlib_dir)

    provenance = json.loads(
        (qlib_dir / "metadata" / "provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["lineage_verified"] is True
    assert len(provenance["dataset_lineage_id"]) == 64
    assert provenance["source_lineage_generation"] == 2


def test_rejects_tampered_snapshot_content(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    source = snapshot / "parquet" / "daily" / "data.parquet"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"original-content")
    relative = source.relative_to(snapshot).as_posix()
    manifest = {
        "name": "snapshot",
        "datasets": {
            "daily": {
                "rows": 1,
                "source_sha256": "a" * 64,
                "files": [
                    {
                        "path": relative,
                        "bytes": source.stat().st_size,
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    }
                ],
            }
        },
    }
    (snapshot / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    source.write_bytes(b"tampered-content")

    with pytest.raises(ValueError, match="snapshot file (size|digest) mismatch"):
        QlibBuilder(snapshot)._write_provenance(tmp_path / "qlib")


def test_failed_qlib_dump_removes_partial_output(tmp_path: Path, monkeypatch) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "manifest.json").write_text(
        json.dumps(
            {
                "name": "snapshot",
                "datasets": {
                    "empty": {"rows": 0, "source_sha256": "a" * 64, "files": []}
                },
            }
        ),
        encoding="utf-8",
    )
    qlib_repo = tmp_path / "qlib-repo"
    script = qlib_repo / "scripts" / "dump_bin.py"
    script.parent.mkdir(parents=True)
    script.write_text("# fixture", encoding="utf-8")
    output = tmp_path / "qlib-output"

    def fail_dump(_command, *, check):
        assert check is True
        output.mkdir()
        (output / "partial.bin").write_bytes(b"partial")
        raise subprocess.CalledProcessError(1, "dump_bin")

    monkeypatch.setattr(subprocess, "run", fail_dump)
    with pytest.raises(subprocess.CalledProcessError):
        QlibBuilder(snapshot).dump_bin(
            staging_by_symbol=tmp_path / "staging",
            qlib_dir=output,
            qlib_repo=qlib_repo,
            qlib_python="python",
            wsl_distro="Ubuntu-22.04",
        )
    assert not output.exists()
