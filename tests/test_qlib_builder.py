import hashlib
import json
import logging
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from quant_data.availability import (
    EVIDENCE_RECOVERABILITY_LEVELS,
    METADATA_AVAILABILITY_LAG_DAYS,
    recoverability_level,
)
from quant_data.qlib_builder import (
    DAILY_QLIB_FIELD_CONTRACT_VERSION,
    QlibBuilder,
    _to_wsl_path,
)

pytestmark = pytest.mark.no_database


def _write_market_control_snapshot(
    tmp_path: Path,
    *,
    ts_code: str,
    up_limit: float,
    down_limit: float,
    include_research_inputs: bool = True,
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
            "amount": 100.0,
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
    if include_research_inputs:
        _write_required_research_inputs(snapshot)
    return snapshot


def _write_required_research_inputs(snapshot: Path) -> None:
    fixtures = {
        "daily_basic": {
            "ts_code": "000001.SZ",
            "trade_date": "2024-01-02",
            "total_mv": 100_000.0,
        },
        "fina_indicator": {
            "ts_code": "000001.SZ",
            "ann_date": "2024-01-01",
            "end_date": "2023-12-31",
            "roe": 10.0,
        },
        "index_member_all": {
            "ts_code": "000001.SZ",
            "l1_code": "801780.SI",
            "in_date": "2021-01-01",
            "out_date": None,
        },
        "index_weight": {
            "index_code": "000300.SH",
            "con_code": "000001.SZ",
            "trade_date": "2024-01-02",
            "weight": 4.5,
        },
        "stock_basic": {
            "ts_code": "000001.SZ",
            "list_date": "2020-01-02",
            "delist_date": None,
        },
        "balancesheet": {
            "ts_code": "000001.SZ",
            "ann_date": "2024-01-01",
            "total_hldr_eqy_exc_min_int": 10_000_000.0,
        },
        "fina_audit": {
            "ts_code": "000001.SZ",
            "ann_date": "2024-01-01",
            "audit_result": "standard_unqualified",
        },
        "namechange": {
            "ts_code": "000001.SZ",
            "name": "平安银行",
            "start_date": "2020-01-02",
            "end_date": None,
        },
    }
    for dataset, row in fixtures.items():
        root = snapshot / "parquet" / dataset
        if root.exists() and any(root.rglob("*.parquet")):
            continue
        target = root / "partition_year=2024"
        target.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_parquet(target / "research.parquet")


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

    _write_required_research_inputs(snapshot)

    by_symbol = QlibBuilder(snapshot).build_staging(tmp_path / "staging")
    frame = pd.read_parquet(by_symbol / "SZ000001.parquet")
    assert frame["symbol"].tolist() == ["SZ000001", "SZ000001"]
    assert frame["close"].tolist() == pytest.approx([1.0, 1.0])
    assert frame["factor"].tolist() == pytest.approx([0.1, 0.2])
    assert frame["volume"].tolist() == pytest.approx([100_000.0, 25_000.0])
    assert frame["vwap"].tolist() == pytest.approx([1.0, 1.0])
    assert frame["up_limit"].tolist() == pytest.approx([1.1, 1.1])
    assert frame["down_limit"].tolist() == pytest.approx([0.9, 0.9])
    # amount is CNY yuan: source 100/25 thousand-CNY becomes 100_000/25_000 yuan,
    # so amount / (hands x 100 shares) stays at the raw price scale.
    assert frame["amount"].tolist() == pytest.approx([100_000.0, 25_000.0])
    raw_hands = pd.Series([100.0, 50.0])
    assert (frame["amount"] / (raw_hands * 100)).tolist() == pytest.approx([10.0, 5.0])


def test_adds_point_in_time_research_features_without_announcement_leakage(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    rows = [
        {
            "ts_code": "000001.SZ",
            "trade_date": day,
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.0,
            "vol": 100.0,
            "amount": 100.0,
            "pct_chg": 0.0,
        }
        for day in ("2024-01-02", "2024-01-03", "2024-01-04")
    ]
    fixtures = {
        "daily": rows,
        "adj_factor": [
            {"ts_code": "000001.SZ", "trade_date": row["trade_date"], "adj_factor": 1.0}
            for row in rows
        ],
        "stk_limit": [
            {
                "ts_code": "000001.SZ",
                "trade_date": row["trade_date"],
                "up_limit": 11.0,
                "down_limit": 9.0,
            }
            for row in rows
        ],
        "daily_basic": [
            {
                "ts_code": "000001.SZ",
                "trade_date": row["trade_date"],
                "turnover_rate": value,
                "pe_ttm": 8.0 + value,
                "pb": 1.0 + value / 10.0,
                "total_mv": 100_000.0,
            }
            for row, value in zip(rows, (1.0, 2.0, 3.0), strict=True)
        ],
        "fina_indicator": [
            {
                "ts_code": "000001.SZ",
                "ann_date": "2024-01-03",
                "end_date": "2023-12-31",
                "roe": 12.5,
                "debt_to_assets": 45.0,
                "netprofit_yoy": 18.0,
            }
        ],
    }
    for dataset, data in fixtures.items():
        target = snapshot / "parquet" / dataset / "partition_year=2024"
        target.mkdir(parents=True)
        pd.DataFrame(data).to_parquet(target / "data.parquet")

    _write_required_research_inputs(snapshot)

    builder = QlibBuilder(snapshot)
    by_symbol = builder.build_staging(tmp_path / "staging")
    frame = pd.read_parquet(by_symbol / "SZ000001.parquet")

    assert frame["turnover_rate"].tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert frame["pe_ttm"].tolist() == pytest.approx([9.0, 10.0, 11.0])
    assert frame["fund_roe"].iloc[:2].isna().all()
    assert frame["fund_roe"].iloc[2] == pytest.approx(12.5)
    assert frame["fund_debt_to_assets"].iloc[2] == pytest.approx(45.0)
    assert frame["fund_netprofit_yoy"].iloc[2] == pytest.approx(18.0)
    lag_label = f"effective_date_with_lag(days={METADATA_AVAILABILITY_LAG_DAYS})"
    assert builder.research_feature_contract["availability_policy"] == {
        "daily_basic": "same_trade_date_after_close",
        "fina_indicator": "strictly_after_announcement_date",
        "income": "strictly_after_announcement_date",
        "balancesheet": "strictly_after_announcement_date",
        "cashflow": "strictly_after_announcement_date",
        "index_weight": lag_label,
        "index_member_all": lag_label,
    }
    assert builder.research_feature_contract["recoverability"] == {
        "daily_basic": "native_history",
        "fina_indicator": "native_history",
        "income": "native_history",
        "balancesheet": "native_history",
        "cashflow": "native_history",
        "index_weight": "native_history",
        "index_member_all": "reconstructed",
    }
    assert "fund_roe" in builder.qlib_fields


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


def test_rejects_qlib_build_without_financial_industry_and_weights(
    tmp_path: Path,
) -> None:
    snapshot = _write_market_control_snapshot(
        tmp_path,
        ts_code="000001.SZ",
        up_limit=11.0,
        down_limit=9.0,
        include_research_inputs=False,
    )

    with pytest.raises(RuntimeError) as raised:
        QlibBuilder(snapshot).build_staging(tmp_path / "staging")

    message = str(raised.value)
    assert "missing daily_basic" in message
    assert "missing fina_indicator" in message
    assert "missing index_member_all" in message
    assert "missing index_weight" in message


def test_rejects_incomplete_historical_benchmark_industry_coverage(
    tmp_path: Path,
) -> None:
    snapshot = _write_market_control_snapshot(
        tmp_path,
        ts_code="000001.SZ",
        up_limit=11.0,
        down_limit=9.0,
    )
    weight_path = (
        snapshot
        / "parquet"
        / "index_weight"
        / "partition_year=2024"
        / "research.parquet"
    )
    pd.DataFrame(
        [
            {
                "index_code": "000300.SH",
                "con_code": instrument,
                "trade_date": "2024-01-02",
                "weight": weight,
            }
            for instrument, weight in (
                ("000001.SZ", 4.5),
                ("600000.SH", 3.5),
            )
        ]
    ).to_parquet(weight_path)

    with pytest.raises(
        RuntimeError,
        match=(
            "no active point-in-time industry for 1/2 "
            "000300.SH constituent-date rows"
        ),
    ):
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
                "amount": 10,
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

    _write_required_research_inputs(snapshot)

    by_symbol = QlibBuilder(snapshot).build_staging(tmp_path / "staging")
    index = pd.read_parquet(by_symbol / "SH000300.parquet")
    assert index["close"].tolist() == pytest.approx([1.0])
    assert index["vwap"].tolist() == pytest.approx([1.0])
    assert index["volume"].tolist() == [0.0]
    assert index["paused"].tolist() == [0.0]
    assert index["amount"].tolist() == pytest.approx([30_000_000.0])


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
                "circ_mv": 100000.0,
            }
        ]
    ).to_parquet(style_source / "styles.parquet")
    qlib_dir = tmp_path / "qlib"
    daily_source = snapshot / "parquet" / "daily"
    daily_source.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "2024-01-31",
                "amount": 600_000.0,
                "vol": 100.0,
            }
        ]
    ).to_parquet(daily_source / "daily.parquet")
    _write_required_research_inputs(snapshot)

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
    # Extended Barra-style schema: the standardized style columns are present
    # alongside the backward-compatible raw log_market_cap.
    for column in (
        "size",
        "nonlinear_size",
        "value",
        "momentum",
        "volatility",
        "liquidity",
        "growth",
        "profitability",
        "leverage",
    ):
        assert column in styles.columns
    # A single-stock cross-section standardizes to the (zero) weighted mean.
    assert styles.loc[0, "size"] == pytest.approx(0.0)
    # The fina_indicator fixture (ann_date 2024-01-01, strictly before the
    # trade date) feeds the profitability descriptor.
    assert styles.loc[0, "profitability"] == pytest.approx(0.0)
    # No adjusted-close history in this fixture: market-derived descriptors
    # stay honestly NaN instead of being fabricated.
    assert pd.isna(styles.loc[0, "momentum"])
    full_market = pd.read_parquet(
        qlib_dir / "metadata" / "full_market_weights.parquet"
    )
    assert full_market.loc[0, "instrument"] == "SZ000001"
    assert full_market.loc[0, "weight"] == pytest.approx(1.0)
    eligibility = pd.read_parquet(qlib_dir / "metadata" / "eligibility_matrix.parquet")
    assert eligibility.loc[0, "instrument"] == "SZ000001"


def test_writes_future_known_trading_calendar_without_future_market_bars(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = tmp_path / "snapshot"
    source = snapshot / "parquet" / "trade_cal" / "partition_year=2024"
    source.mkdir(parents=True)
    pd.DataFrame(
        [
            {"cal_date": "2024-01-02", "is_open": 1},
            {"cal_date": "2024-01-03", "is_open": 1},
            {"cal_date": "2024-01-04", "is_open": 0},
        ]
    ).to_parquet(source / "calendar.parquet", index=False)
    builder = QlibBuilder(snapshot)
    monkeypatch.setattr(builder, "_write_market_context_metadata", lambda _: False)
    monkeypatch.setattr(builder, "_write_eligibility_metadata", lambda _: True)

    qlib_dir = tmp_path / "qlib"
    builder._write_portfolio_metadata(qlib_dir)

    calendar = pd.read_parquet(
        qlib_dir / "metadata" / "known_trading_calendar.parquet"
    )
    assert calendar["date"].dt.date.astype(str).tolist() == [
        "2024-01-02",
        "2024-01-03",
    ]


def test_writes_normalized_market_context_without_stock_level_duplication(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    global_root = snapshot / "parquet" / "index_global"
    shibor_root = snapshot / "parquet" / "shibor"
    global_root.mkdir(parents=True)
    shibor_root.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ts_code": "SPX",
                "trade_date": "2024-01-02",
                "close": 4750.0,
                "pct_chg": 0.5,
            }
        ]
    ).to_parquet(global_root / "global.parquet")
    pd.DataFrame(
        [{"date": "2024-01-02", "on": 1.75, "1w": 1.82, "1y": 2.10}]
    ).to_parquet(shibor_root / "shibor.parquet")

    target = tmp_path / "qlib" / "metadata"
    assert QlibBuilder(snapshot)._write_market_context_metadata(target) is True

    context = pd.read_parquet(target / "market_context.parquet")
    assert set(context["source"]) == {"index_global", "shibor"}
    assert set(context.loc[context["source"] == "index_global", "instrument"]) == {"SPX"}
    assert set(context.loc[context["source"] == "shibor", "instrument"]) == {"shibor"}
    contract = json.loads(
        (target / "market_context_contract.json").read_text(encoding="utf-8")
    )
    assert contract["sources"]["shibor"]["features"] == ["on", "1w", "1y"]


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
    assert provenance["field_contract_version"] == DAILY_QLIB_FIELD_CONTRACT_VERSION
    assert provenance["source_volume_unit"] == "hand"
    assert provenance["qlib_volume_unit"] == "share"
    assert provenance["source_amount_unit"] == "thousand_cny"
    assert provenance["qlib_amount_unit"] == "cny"
    assert provenance["source_hand_size"] == 100
    assert provenance["index_volume_policy"] == "excluded_non_tradable_benchmark"
    assert provenance["field_units"]["amount"] == "cny_yuan"
    assert provenance["field_units"]["close"] == "snapshot_anchor_normalized_price"
    assert provenance["field_units"]["factor"] == "adj_factor_div_base_price"
    assert provenance["field_units"]["change"] == "decimal_return"
    assert provenance["field_units"]["volume"] == (
        "value_consistent_shares_price_times_volume_equals_cny_amount"
    )
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


def test_rejects_daily_amount_and_hand_volume_with_impossible_vwap(tmp_path: Path) -> None:
    snapshot = _write_market_control_snapshot(
        tmp_path,
        ts_code="000001.SZ",
        up_limit=11.0,
        down_limit=9.0,
    )
    daily = next((snapshot / "parquet" / "daily").rglob("*.parquet"))
    frame = pd.read_parquet(daily)
    frame["amount"] = 1_000.0
    frame.to_parquet(daily, index=False)

    with pytest.raises(RuntimeError, match="hand/amount price contract"):
        QlibBuilder(snapshot).build_staging(tmp_path / "staging")


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


def _write_revision_fixture(
    snapshot: Path,
    fina_rows: list[dict],
    daily_days: tuple[str, ...] = ("2024-01-02",),
) -> None:
    rows = [
        {
            "ts_code": "000001.SZ",
            "trade_date": day,
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.0,
            "vol": 100.0,
            "amount": 100.0,
            "pct_chg": 0.0,
        }
        for day in daily_days
    ]
    fixtures = {
        "daily": rows,
        "adj_factor": [
            {"ts_code": "000001.SZ", "trade_date": row["trade_date"], "adj_factor": 1.0}
            for row in rows
        ],
        "stk_limit": [
            {
                "ts_code": "000001.SZ",
                "trade_date": row["trade_date"],
                "up_limit": 11.0,
                "down_limit": 9.0,
            }
            for row in rows
        ],
        "daily_basic": [
            {
                "ts_code": "000001.SZ",
                "trade_date": row["trade_date"],
                "total_mv": 100_000.0,
            }
            for row in rows
        ],
        "fina_indicator": fina_rows,
    }
    for dataset, data in fixtures.items():
        target = snapshot / "parquet" / dataset / "partition_year=2024"
        target.mkdir(parents=True)
        pd.DataFrame(data).to_parquet(target / "data.parquet")
    _write_required_research_inputs(snapshot)


def _fund_roe(by_symbol: Path) -> list[float]:
    return pd.read_parquet(by_symbol / "SZ000001.parquet")["fund_roe"].tolist()


def test_fundamental_revision_conflict_prefers_newest_f_ann_date(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    _write_revision_fixture(
        snapshot,
        [
            {
                "ts_code": "000001.SZ",
                "ann_date": "2024-01-01",
                "end_date": "2023-12-31",
                "roe": 10.0,
                "f_ann_date": "2024-01-01",
                "update_flag": 0,
            },
            {
                "ts_code": "000001.SZ",
                "ann_date": "2024-01-01",
                "end_date": "2023-12-31",
                "roe": 20.0,
                "f_ann_date": "2024-01-05",
                "update_flag": 1,
            },
        ],
    )

    by_symbol = QlibBuilder(snapshot).build_staging(tmp_path / "staging")

    # Same (ts_code, ann_date, end_date) conflict: the revision with the newer
    # f_ann_date / update_flag wins deterministically.
    assert _fund_roe(by_symbol) == pytest.approx([20.0])


def test_fundamental_revision_conflict_uses_latest_ingested_at(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    _write_revision_fixture(
        snapshot,
        [
            {
                "ts_code": "000001.SZ",
                "ann_date": "2024-01-01",
                "end_date": "2023-12-31",
                "roe": 10.0,
                "ingested_at": pd.Timestamp("2026-01-01T00:00:00Z"),
            },
            {
                "ts_code": "000001.SZ",
                "ann_date": "2024-01-01",
                "end_date": "2023-12-31",
                "roe": 20.0,
                "ingested_at": pd.Timestamp("2026-06-01T00:00:00Z"),
            },
        ],
    )

    by_symbol = QlibBuilder(snapshot).build_staging(tmp_path / "staging")

    assert _fund_roe(by_symbol) == pytest.approx([20.0])


def test_fundamental_revision_dedup_is_deterministic_across_row_order(
    tmp_path: Path,
) -> None:
    base_rows = [
        {
            "ts_code": "000001.SZ",
            "ann_date": "2024-01-01",
            "end_date": "2023-12-31",
            "roe": 10.0,
        },
        {
            "ts_code": "000001.SZ",
            "ann_date": "2024-01-01",
            "end_date": "2023-12-31",
            "roe": 20.0,
        },
    ]
    frames = []
    for name, rows in (("forward", base_rows), ("reversed", list(reversed(base_rows)))):
        snapshot = tmp_path / name / "snapshot"
        _write_revision_fixture(snapshot, rows)
        by_symbol = QlibBuilder(snapshot).build_staging(tmp_path / name / "staging")
        frames.append(pd.read_parquet(by_symbol / "SZ000001.parquet"))

    # With no revision columns at all, the content-hash tie-break still makes
    # the surviving row independent of parquet row order.
    pd.testing.assert_frame_equal(frames[0], frames[1])

    # Rebuilding the same snapshot twice produces the identical frame.
    snapshot = tmp_path / "forward" / "snapshot"
    again = QlibBuilder(snapshot).build_staging(tmp_path / "forward" / "staging-again")
    pd.testing.assert_frame_equal(frames[0], pd.read_parquet(again / "SZ000001.parquet"))


def test_financial_restatement_applies_only_after_the_new_announcement(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    _write_revision_fixture(
        snapshot,
        [
            {
                "ts_code": "000001.SZ",
                "ann_date": "2024-01-01",
                "end_date": "2023-12-31",
                "roe": 10.0,
            },
            {
                "ts_code": "000001.SZ",
                "ann_date": "2024-01-05",
                "end_date": "2023-12-31",
                "roe": 99.0,
            },
        ],
        daily_days=("2024-01-02", "2024-01-03", "2024-01-04", "2024-01-08"),
    )

    by_symbol = QlibBuilder(snapshot).build_staging(tmp_path / "staging")

    # Restatement scenario: the revised value is invisible before its own
    # announcement date and fully replaces the old one afterwards.
    assert _fund_roe(by_symbol) == pytest.approx([10.0, 10.0, 10.0, 99.0])

    rebuilt = QlibBuilder(snapshot).build_staging(tmp_path / "staging-again")
    assert _fund_roe(rebuilt) == pytest.approx([10.0, 10.0, 10.0, 99.0])


def test_research_contract_admits_only_evidence_grade_recoverability(
    tmp_path: Path,
) -> None:
    snapshot = _write_market_control_snapshot(
        tmp_path, ts_code="000001.SZ", up_limit=11.0, down_limit=9.0
    )

    contract = QlibBuilder(snapshot).research_feature_contract

    # current_only / unavailable datasets must never feed formal evidence
    # features or their point-in-time metadata (design draft 3.3).
    assert contract["availability_policy"]
    for dataset in contract["availability_policy"]:
        assert recoverability_level(dataset) in EVIDENCE_RECOVERABILITY_LEVELS


_EXTENDED_FINA_ROW = {
    "ts_code": "000001.SZ",
    "ann_date": "2024-01-03",
    "end_date": "2023-12-31",
    "roe": 12.5,
    "eps": 0.85,
    "bps": 6.4,
    "ocfps": 1.1,
    "roe_waa": 13.0,
    "roe_dt": 12.1,
    "roic": 9.5,
    "netprofit_margin": 21.0,
    "assets_turn": 0.8,
    "inv_turn": 5.2,
    "ar_turn": 7.6,
    "quick_ratio": 1.3,
    "debt_to_eqt": 80.0,
    "saleexp_to_gr": 4.5,
    "adminexp_of_gr": 6.5,
    "finaexp_of_gr": 1.2,
    "op_yoy": 15.0,
    "equity_yoy": 8.0,
    "ocf_to_or": 0.18,
    "ocf_to_profit": 1.05,
    "salescash_to_or": 1.12,
    "interestdebt": 3.5e8,
}

_EXTENDED_FUND_TARGETS = {
    "fund_eps": 0.85,
    "fund_bps": 6.4,
    "fund_ocfps": 1.1,
    "fund_roe_weighted": 13.0,
    "fund_roe_diluted": 12.1,
    "fund_roic": 9.5,
    "fund_netprofit_margin": 21.0,
    "fund_assets_turnover": 0.8,
    "fund_inventory_turnover": 5.2,
    "fund_receivables_turnover": 7.6,
    "fund_quick_ratio": 1.3,
    "fund_debt_to_equity": 80.0,
    "fund_sales_expense_ratio": 4.5,
    "fund_admin_expense_ratio": 6.5,
    "fund_finance_expense_ratio": 1.2,
    "fund_op_profit_yoy": 15.0,
    "fund_equity_yoy": 8.0,
    "fund_ocf_to_revenue": 0.18,
    "fund_ocf_to_profit": 1.05,
    "fund_sales_cash_to_revenue": 1.12,
    "fund_interest_debt": 3.5e8,
}


def test_extended_fina_indicator_fields_enter_bin_point_in_time(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    _write_revision_fixture(
        snapshot,
        [dict(_EXTENDED_FINA_ROW)],
        daily_days=("2024-01-02", "2024-01-03", "2024-01-04"),
    )

    builder = QlibBuilder(snapshot)
    by_symbol = builder.build_staging(tmp_path / "staging")
    frame = pd.read_parquet(by_symbol / "SZ000001.parquet")

    # Every extended field lands in the binary staging frame and is invisible
    # before its announcement date (ann_date 2024-01-03 -> first value on
    # the strictly later trade date 2024-01-04).
    for target, value in _EXTENDED_FUND_TARGETS.items():
        assert target in builder.qlib_fields
        assert frame[target].iloc[:2].isna().all()
        assert frame[target].iloc[2] == pytest.approx(value)

    units = builder._field_units()
    assert units["fund_eps"] == "cny_yuan_per_share"
    assert units["fund_bps"] == "cny_yuan_per_share"
    assert units["fund_ocfps"] == "cny_yuan_per_share"
    assert units["fund_roic"] == "percent"
    assert units["fund_assets_turnover"] == "turnover_times"
    assert units["fund_quick_ratio"] == "ratio_unitless"
    assert units["fund_interest_debt"] == "cny_yuan"


def test_contract_distinguishes_missing_from_all_null_source_columns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    snapshot = tmp_path / "snapshot"
    _write_revision_fixture(
        snapshot,
        [
            {
                "ts_code": "000001.SZ",
                "ann_date": "2024-01-03",
                "end_date": "2023-12-31",
                "roe": 12.5,
                # Declared and present in the schema, but entirely null.
                "ocf_to_or": None,
            }
        ],
    )

    with caplog.at_level(logging.WARNING, logger="quant_data.qlib_builder"):
        builder = QlibBuilder(snapshot)

    contract = builder.research_feature_contract
    assert contract["version"] == 4
    # The Tushare non-default columns the downloader never receives are
    # reported as missing source columns instead of being silently skipped.
    missing = contract["missing_fundamental_fields"]["fina_indicator"]
    assert missing["q_profit_yoy"] == "fund_quarter_profit_yoy"
    assert "ocf_to_or" not in missing
    # A column that exists but holds no non-null value is a distinct class.
    assert contract["all_null_fundamental_fields"]["fina_indicator"] == {
        "ocf_to_or": "fund_ocf_to_revenue"
    }
    # Missing sources stay out of the injected fields; all-null sources keep
    # their (all-NaN) channel, matching the accepted NaN-channel semantics
    # for values an issuer never discloses.
    assert "fund_quarter_profit_yoy" not in contract["fields"]
    assert "fund_ocf_to_revenue" in contract["fields"]
    # Both drift classes surface as build-time warnings without blocking.
    messages = [record.getMessage() for record in caplog.records]
    assert any("absent" in message for message in messages)
    assert any("only null" in message for message in messages)

    by_symbol = builder.build_staging(tmp_path / "staging")
    frame = pd.read_parquet(by_symbol / "SZ000001.parquet")
    assert "fund_quarter_profit_yoy" not in frame.columns
    assert frame["fund_ocf_to_revenue"].isna().all()


def _write_statement_fixture(
    snapshot: Path,
    *,
    income_rows: list[dict],
    balancesheet_rows: list[dict],
    cashflow_rows: list[dict],
    daily_days: tuple[str, ...] = ("2024-01-02",),
) -> None:
    # Statement parquet files must be written before _write_revision_fixture so
    # the default research inputs do not create a second, differently-schemad
    # balancesheet parquet in the same dataset directory.
    for dataset, rows in (
        ("income", income_rows),
        ("balancesheet", balancesheet_rows),
        ("cashflow", cashflow_rows),
    ):
        target = snapshot / "parquet" / dataset / "partition_year=2024"
        target.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(target / "data.parquet")
    _write_revision_fixture(
        snapshot,
        [
            {
                "ts_code": "000001.SZ",
                "ann_date": "2024-01-01",
                "end_date": "2023-12-31",
                "roe": 10.0,
            }
        ],
        daily_days=daily_days,
    )


def test_statement_line_items_follow_announcement_dates_without_lookahead(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    _write_statement_fixture(
        snapshot,
        income_rows=[
            {
                "ts_code": "000001.SZ",
                "ann_date": "2024-01-01",
                "end_date": "2023-12-31",
                "n_income_attr_p": 1.0e6,
                "rd_exp": 5.0e4,
            },
            {
                "ts_code": "000001.SZ",
                "ann_date": "2024-01-05",
                "end_date": "2023-12-31",
                "n_income_attr_p": 2.0e6,
                "rd_exp": 6.0e4,
            },
        ],
        balancesheet_rows=[
            {
                "ts_code": "000001.SZ",
                "ann_date": "2024-01-02",
                "end_date": "2023-12-31",
                "total_assets": 5.0e7,
                "money_cap": 8.0e6,
                "goodwill": 1.0e6,
            }
        ],
        cashflow_rows=[
            {
                "ts_code": "000001.SZ",
                "ann_date": "2024-01-03",
                "end_date": "2023-12-31",
                "n_cashflow_act": 3.0e6,
                "c_pay_acq_const_fiolta": 9.0e5,
            }
        ],
        daily_days=("2024-01-02", "2024-01-03", "2024-01-04", "2024-01-08"),
    )

    builder = QlibBuilder(snapshot)
    by_symbol = builder.build_staging(tmp_path / "staging")
    frame = pd.read_parquet(by_symbol / "SZ000001.parquet")

    # Income restatement: the revised value is invisible before its own
    # announcement date (2024-01-05) and fully replaces the old one after.
    assert frame["fund_net_profit"].tolist() == pytest.approx(
        [1.0e6, 1.0e6, 1.0e6, 2.0e6]
    )
    assert frame["fund_rd_expense"].tolist() == pytest.approx(
        [5.0e4, 5.0e4, 5.0e4, 6.0e4]
    )
    # Balance sheet announced 2024-01-02: visible strictly after that date.
    assert pd.isna(frame["fund_total_assets"].iloc[0])
    assert frame["fund_total_assets"].iloc[1:].tolist() == pytest.approx([5.0e7] * 3)
    assert frame["fund_money_cap"].iloc[1:].tolist() == pytest.approx([8.0e6] * 3)
    assert frame["fund_goodwill"].iloc[1:].tolist() == pytest.approx([1.0e6] * 3)
    # Cash flow announced 2024-01-03: visible from 2024-01-04 onwards.
    assert frame["fund_ocf_net"].iloc[:2].isna().all()
    assert frame["fund_ocf_net"].iloc[2:].tolist() == pytest.approx([3.0e6] * 2)
    assert frame["fund_capex"].iloc[:2].isna().all()
    assert frame["fund_capex"].iloc[2:].tolist() == pytest.approx([9.0e5] * 2)

    for target in (
        "fund_net_profit",
        "fund_rd_expense",
        "fund_total_assets",
        "fund_money_cap",
        "fund_goodwill",
        "fund_ocf_net",
        "fund_capex",
    ):
        assert target in builder.qlib_fields
    units = builder._field_units()
    assert units["fund_net_profit"] == "cny_yuan"
    assert units["fund_total_assets"] == "cny_yuan"
    assert units["fund_ocf_net"] == "cny_yuan"


def test_statement_revision_conflict_prefers_newest_update_flag(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    _write_statement_fixture(
        snapshot,
        income_rows=[
            {
                "ts_code": "000001.SZ",
                "ann_date": "2024-01-01",
                "end_date": "2023-12-31",
                "n_income_attr_p": 1.0e6,
                "f_ann_date": "2024-01-01",
                "update_flag": 0,
            },
            {
                "ts_code": "000001.SZ",
                "ann_date": "2024-01-01",
                "end_date": "2023-12-31",
                "n_income_attr_p": 2.0e6,
                "f_ann_date": "2024-01-02",
                "update_flag": 1,
            },
        ],
        balancesheet_rows=[
            {
                "ts_code": "000001.SZ",
                "ann_date": "2024-01-01",
                "end_date": "2023-12-31",
                "total_assets": 5.0e7,
            }
        ],
        cashflow_rows=[
            {
                "ts_code": "000001.SZ",
                "ann_date": "2024-01-01",
                "end_date": "2023-12-31",
                "n_cashflow_act": 3.0e6,
            }
        ],
    )

    by_symbol = QlibBuilder(snapshot).build_staging(tmp_path / "staging")
    frame = pd.read_parquet(by_symbol / "SZ000001.parquet")

    # Same (ts_code, ann_date, end_date) conflict in the income statement: the
    # revision with the newer f_ann_date / update_flag wins deterministically.
    assert frame["fund_net_profit"].tolist() == pytest.approx([2.0e6])
