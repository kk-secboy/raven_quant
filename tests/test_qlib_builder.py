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
from quant_data.history_bounds import (
    BSE_GOVERNED_HISTORY_START,
    GOVERNED_DAILY_STOCK_SCOPE_VERSION,
)
from quant_data.qlib_builder import (
    _MAX_EXCLUDED_DAILY_UNIT_RATIO,
    DAILY_QLIB_DUCKDB_MEMORY_LIMIT,
    DAILY_QLIB_DUCKDB_THREADS,
    DAILY_QLIB_FIELD_CONTRACT_VERSION,
    QlibBuilder,
    _to_wsl_path,
    build_qlib_output_manifest,
    verify_qlib_output_manifest,
)
from quant_data.snapshot_lineage import make_lineage_id
from quant_data.universe import (
    GOVERNED_DAILY_ETF_WHITELIST,
    governed_daily_etf_whitelist_contract,
)

pytestmark = pytest.mark.no_database


def test_daily_duckdb_connection_applies_host_safe_resource_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_connect = __import__("duckdb").connect
    statements: list[str] = []

    class RecordingConnection:
        def __init__(self) -> None:
            self.connection = real_connect()

        def execute(self, query: str):
            statements.append(query)
            return self.connection.execute(query)

        def close(self) -> None:
            self.connection.close()

    monkeypatch.setattr(
        "quant_data.qlib_builder.duckdb.connect",
        lambda: RecordingConnection(),
    )
    spill = tmp_path / "daily-spill"

    connection = QlibBuilder._duckdb_connection(spill_dir=spill)
    connection.close()

    normalized = [statement.replace("\\", "/") for statement in statements]
    assert f"SET memory_limit='{DAILY_QLIB_DUCKDB_MEMORY_LIMIT}'" in statements
    assert f"SET threads={DAILY_QLIB_DUCKDB_THREADS}" in statements
    assert "SET preserve_insertion_order=false" in statements
    assert f"SET temp_directory='{spill.resolve().as_posix()}'" in normalized


def test_style_metadata_symbol_batches_match_full_panel_math(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "snapshot"
    rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    factor_rows: list[dict[str, object]] = []
    for symbol, market_cap, close in (
        ("000001.SZ", 100_000.0, 10.0),
        ("600000.SH", 400_000.0, 20.0),
    ):
        for trade_date, multiplier in (
            ("2023-12-29", 1.0),
            ("2024-01-02", 1.01),
        ):
            rows.append(
                {
                    "ts_code": symbol,
                    "trade_date": trade_date,
                    "total_mv": market_cap * multiplier,
                    "circ_mv": market_cap * multiplier * 0.8,
                    "pb": 2.0,
                    "pe_ttm": 10.0,
                    "turnover_rate": 1.0,
                }
            )
            daily_rows.append(
                {
                    "ts_code": symbol,
                    "trade_date": trade_date,
                    "close": close * multiplier,
                }
            )
            factor_rows.append(
                {
                    "ts_code": symbol,
                    "trade_date": trade_date,
                    "adj_factor": 1.0,
                }
            )
    for symbol, trade_date, market_cap, close in (
        ("200001.SZ", "2024-01-02", 900_000.0, 5.0),
        ("201872.SZ", "2024-01-02", 800_000.0, 6.0),
        ("920001.BJ", "2022-12-30", 700_000.0, 7.0),
        ("920001.BJ", "2023-01-03", 300_000.0, 8.0),
    ):
        rows.append(
            {
                "ts_code": symbol,
                "trade_date": trade_date,
                "total_mv": market_cap,
                "circ_mv": market_cap * 0.8,
                "pb": 2.0,
                "pe_ttm": 10.0,
                "turnover_rate": 1.0,
            }
        )
        daily_rows.append(
            {"ts_code": symbol, "trade_date": trade_date, "close": close}
        )
        factor_rows.append(
            {"ts_code": symbol, "trade_date": trade_date, "adj_factor": 1.0}
        )
    for dataset, frame in (
        ("daily_basic", pd.DataFrame(rows)),
        ("daily", pd.DataFrame(daily_rows)),
        ("adj_factor", pd.DataFrame(factor_rows)),
        (
            "fina_indicator",
            pd.DataFrame(
                [
                    {
                        "ts_code": symbol,
                        "ann_date": "2023-01-01",
                        "roe": 10.0,
                        "or_yoy": 5.0,
                        "netprofit_yoy": 6.0,
                        "debt_to_assets": 40.0,
                    }
                    for symbol in ("000001.SZ", "600000.SH", "920001.BJ")
                ]
            ),
        ),
        (
            "stock_basic",
            pd.DataFrame(
                [
                    {
                        "ts_code": symbol,
                        "list_date": "2020-01-02",
                        "delist_date": None,
                    }
                    for symbol in ("000001.SZ", "200001.SZ", "600000.SH", "920001.BJ")
                ]
            ),
        ),
    ):
        root = snapshot / "parquet" / dataset
        root.mkdir(parents=True)
        frame.to_parquet(root / "data.parquet", index=False)

    builder = QlibBuilder(snapshot)
    expected = builder._build_style_exposures(pd.DataFrame(rows))
    calls: list[tuple[str, tuple[str, ...]]] = []
    original_read = builder._read_dataset_for_symbols

    def recording_read(
        dataset: str,
        columns,
        symbols,
        *,
        required=(),
    ):
        calls.append((dataset, tuple(symbols)))
        return original_read(dataset, columns, symbols, required=required)

    monkeypatch.setattr(
        "quant_data.qlib_builder.DAILY_QLIB_STYLE_SYMBOL_BATCH", 1
    )
    monkeypatch.setattr(builder, "_read_dataset_for_symbols", recording_read)
    target = tmp_path / "metadata"

    assert builder._write_style_metadata_bounded(target) is True

    actual = pd.read_parquet(target / "style_exposures.parquet")
    columns = list(expected.columns)
    pd.testing.assert_frame_equal(
        actual.loc[:, columns].sort_values(["datetime", "instrument"]).reset_index(drop=True),
        expected.loc[:, columns]
        .sort_values(["datetime", "instrument"])
        .reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    daily_basic_batches = [symbols for dataset, symbols in calls if dataset == "daily_basic"]
    assert daily_basic_batches == [("000001.SZ",), ("600000.SH",), ("920001.BJ",)]
    assert all(len(symbols) == 1 for _, symbols in calls)
    weights = pd.read_parquet(target / "full_market_weights.parquet")
    assert weights.groupby("datetime")["weight"].sum().tolist() == pytest.approx(
        [1.0, 1.0, 1.0]
    )
    assert set(weights["instrument"]) == {"BJ920001", "SH600000", "SZ000001"}
    assert pd.to_datetime(
        weights.loc[weights["instrument"].eq("BJ920001"), "datetime"]
    ).dt.date.min() >= BSE_GOVERNED_HISTORY_START
    assert not (target / ".style_metadata_attempt").exists()


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
        "moneyflow": {
            "ts_code": "000001.SZ",
            "trade_date": "2024-01-02",
            "net_mf_amount": 5.0,
            "buy_lg_amount": 30.0,
            "sell_lg_amount": 20.0,
            "buy_elg_amount": 10.0,
            "sell_elg_amount": 10.0,
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


def _write_governed_etf_inputs(snapshot: Path, *, include_factor: bool = True) -> None:
    fixtures = {
        "fund_daily": pd.DataFrame(
            [
                {
                    "ts_code": "510300.SH",
                    "trade_date": "2024-01-02",
                    "open": 4.0,
                    "high": 4.1,
                    "low": 3.9,
                    "close": 4.0,
                    "pre_close": 4.0,
                    "change": 0.0,
                    "pct_chg": 0.0,
                    "vol": 100.0,
                    "amount": 40.0,
                },
                {
                    "ts_code": "510300.SH",
                    "trade_date": "2024-01-03",
                    "open": 4.2,
                    "high": 4.4,
                    "low": 4.1,
                    "close": 4.4,
                    "pre_close": 4.0,
                    "change": 0.4,
                    "pct_chg": 10.0,
                    "vol": 100.0,
                    "amount": 44.0,
                },
                # fund_daily can also contain LOFs/non-whitelisted funds.  They
                # must never enter the Qlib investable surface by prefix alone.
                {
                    "ts_code": "160105.SZ",
                    "trade_date": "2024-01-02",
                    "open": 1.0,
                    "high": 1.0,
                    "low": 1.0,
                    "close": 1.0,
                    "pre_close": 1.0,
                    "change": 0.0,
                    "pct_chg": 0.0,
                    "vol": 100.0,
                    "amount": 10.0,
                },
            ]
        ),
        "fund_basic": pd.DataFrame(
            [
                {
                    "ts_code": "510300.SH",
                    "market": "E",
                    "list_date": "20120528",
                    "delist_date": None,
                }
            ]
        ),
    }
    if include_factor:
        fixtures["fund_adj"] = pd.DataFrame(
            [
                {
                    "ts_code": "510300.SH",
                    "trade_date": "2024-01-02",
                    "adj_factor": 1.0,
                },
                {
                    "ts_code": "510300.SH",
                    "trade_date": "2024-01-03",
                    "adj_factor": 2.0,
                },
                {
                    "ts_code": "160105.SZ",
                    "trade_date": "2024-01-02",
                    "adj_factor": 1.0,
                },
            ]
        )
    for dataset, frame in fixtures.items():
        root = snapshot / "parquet" / dataset / "partition_year=2024"
        root.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(root / "data.parquet", index=False)


def test_governed_etf_staging_is_whitelist_only_and_keeps_financials_null(
    tmp_path: Path,
) -> None:
    snapshot = _write_market_control_snapshot(
        tmp_path, ts_code="000001.SZ", up_limit=11.0, down_limit=9.0
    )
    _write_governed_etf_inputs(snapshot)

    builder = QlibBuilder(snapshot)
    by_symbol = builder.build_staging(tmp_path / "staging")

    etf = pd.read_parquet(by_symbol / "SH510300.parquet")
    assert not (by_symbol / "SZ160105.parquet").exists()
    assert etf["close"].tolist() == pytest.approx([1.0, 2.2])
    assert etf["volume"].tolist() == pytest.approx([40_000.0, 20_000.0])
    assert etf["amount"].tolist() == pytest.approx([40_000.0, 44_000.0])
    assert etf["up_limit"].tolist() == pytest.approx([1.1, 2.2])
    assert etf["down_limit"].tolist() == pytest.approx([0.9, 1.8])
    assert etf["fund_roe"].isna().all()
    evidence = builder._governed_etf_evidence()
    assert evidence["included_symbols"] == ["510300.SH"]
    assert len(evidence["whitelist_sha256"]) == 64
    assert evidence["whitelist_sha256"] == governed_daily_etf_whitelist_contract()[
        "whitelist_sha256"
    ]
    coverage = builder._field_year_coverage_evidence()
    close_year = coverage["fields"]["close"]["years"][0]
    assert close_year["asset_type_rows"]["etf"] == 2
    assert "fund_daily" in close_year["source_contracts"]
    factor_year = coverage["fields"]["factor"]["years"][0]
    assert "fund_adj" in factor_year["source_contracts"]
    limit_year = coverage["fields"]["up_limit"]["years"][0]
    assert any(
        source.startswith("governed_etf_price_limit:")
        for source in limit_year["source_contracts"]
    )

    qlib_dir = tmp_path / "qlib"
    (qlib_dir / "instruments").mkdir(parents=True)
    builder._write_stock_universe(qlib_dir)
    universe = (qlib_dir / "instruments" / "cn_all.txt").read_text(encoding="utf-8")
    assert "SH510300\t2024-01-02\t2024-01-03" in universe
    assert "SZ160105" not in universe


def test_governed_etf_source_fails_closed_without_matching_fund_adj(
    tmp_path: Path,
) -> None:
    snapshot = _write_market_control_snapshot(
        tmp_path, ts_code="000001.SZ", up_limit=11.0, down_limit=9.0
    )
    _write_governed_etf_inputs(snapshot, include_factor=False)

    with pytest.raises(RuntimeError, match="sources are partial; missing fund_adj"):
        QlibBuilder(snapshot).build_staging(tmp_path / "staging")


def test_production_etf_contract_requires_every_whitelisted_symbol(
    tmp_path: Path,
) -> None:
    snapshot = _write_market_control_snapshot(
        tmp_path, ts_code="000001.SZ", up_limit=11.0, down_limit=9.0
    )
    _write_governed_etf_inputs(snapshot)

    with pytest.raises(RuntimeError, match="no fund_daily history"):
        QlibBuilder(snapshot, require_governed_etfs=True).build_staging(
            tmp_path / "staging"
        )


def test_complete_production_etf_whitelist_builds_under_strict_contract(
    tmp_path: Path,
) -> None:
    snapshot = _write_market_control_snapshot(
        tmp_path, ts_code="000001.SZ", up_limit=11.0, down_limit=9.0
    )
    _write_governed_etf_inputs(snapshot)
    for dataset in ("fund_daily", "fund_adj", "fund_basic"):
        path = next((snapshot / "parquet" / dataset).rglob("*.parquet"))
        source = pd.read_parquet(path)
        template = source[source["ts_code"].eq("510300.SH")]
        rows = [template.assign(ts_code=symbol) for symbol in GOVERNED_DAILY_ETF_WHITELIST]
        if dataset == "fund_daily":
            rows.append(source[source["ts_code"].eq("160105.SZ")])
        pd.concat(rows, ignore_index=True).to_parquet(path, index=False)

    builder = QlibBuilder(snapshot, require_governed_etfs=True)
    by_symbol = builder.build_staging(tmp_path / "staging")

    assert builder._governed_etf_evidence()["missing_symbols"] == []
    assert all(
        (by_symbol / f"SH{symbol.split('.', 1)[0]}.parquet").is_file()
        for symbol in GOVERNED_DAILY_ETF_WHITELIST
    )


def test_etf_eligibility_skips_stock_financial_and_st_gates(tmp_path: Path) -> None:
    snapshot = _write_market_control_snapshot(
        tmp_path, ts_code="000001.SZ", up_limit=11.0, down_limit=9.0
    )
    _write_governed_etf_inputs(snapshot)
    target = tmp_path / "metadata"

    assert QlibBuilder(snapshot)._write_eligibility_metadata(target) is True

    matrix = pd.read_parquet(target / "eligibility_matrix.parquet")
    etf = matrix[matrix["instrument"].eq("SH510300")]
    assert set(etf["asset_type"]) == {"etf"}
    assert not etf["financial_gate_required"].any()
    assert etf["equity"].isna().all()
    reasons = [reason for raw in etf["reasons"] for reason in json.loads(raw)]
    assert "negative_or_missing_equity" not in reasons
    assert "nonstandard_or_missing_audit" not in reasons
    assert "st" not in reasons


def test_eligibility_metadata_reads_full_history_in_bounded_symbol_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _write_market_control_snapshot(
        tmp_path, ts_code="000001.SZ", up_limit=11.0, down_limit=9.0
    )
    daily_path = next((snapshot / "parquet" / "daily").rglob("*.parquet"))
    daily = pd.read_parquet(daily_path)
    second = daily.iloc[0].copy()
    second["ts_code"] = "600000.SH"
    second["vol"] = 0.0
    pd.concat([daily, second.to_frame().T], ignore_index=True).to_parquet(
        daily_path, index=False
    )
    for dataset, extra in (
        ("stock_basic", {"ts_code": "600000.SH", "list_date": "2020-01-02"}),
        (
            "balancesheet",
            {
                "ts_code": "600000.SH",
                "ann_date": "2024-01-01",
                "total_hldr_eqy_exc_min_int": 10_000_000.0,
            },
        ),
        (
            "fina_audit",
            {
                "ts_code": "600000.SH",
                "ann_date": "2024-01-01",
                "audit_result": "standard_unqualified",
            },
        ),
        (
            "namechange",
            {
                "ts_code": "600000.SH",
                "name": "*ST浦发",
                "start_date": "2024-01-02",
                "end_date": None,
            },
        ),
    ):
        path = next((snapshot / "parquet" / dataset).rglob("*.parquet"))
        frame = pd.read_parquet(path)
        pd.concat([frame, pd.DataFrame([extra])], ignore_index=True).to_parquet(
            path, index=False
        )
    regulatory_root = snapshot / "parquet" / "regulatory_events"
    regulatory_root.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ts_code": "600000.SH",
                "event_date": "2024-01-01",
                "known_date": "2024-01-02",
                "major": True,
            }
        ]
    ).to_parquet(regulatory_root / "events.parquet", index=False)

    builder = QlibBuilder(snapshot)
    calls: list[tuple[str, tuple[str, ...]]] = []
    original_read = builder._read_dataset_for_symbols

    def recording_read(dataset, columns, symbols, *, required=()):
        calls.append((dataset, tuple(symbols)))
        return original_read(dataset, columns, symbols, required=required)

    monkeypatch.setattr(
        "quant_data.qlib_builder.DAILY_QLIB_ELIGIBILITY_SYMBOL_BATCH", 1
    )
    monkeypatch.setattr(builder, "_read_dataset_for_symbols", recording_read)
    target = tmp_path / "metadata"

    assert builder._write_eligibility_metadata(target) is True

    daily_batches = [symbols for dataset, symbols in calls if dataset == "daily"]
    assert daily_batches == [("000001.SZ",), ("600000.SH",)]
    assert all(len(symbols) == 1 for _, symbols in calls)
    result = pd.read_parquet(target / "eligibility_matrix.parquet").set_index(
        "instrument"
    )
    second_reasons = json.loads(result.loc["SH600000", "reasons"])
    assert result.loc["SH600000", "suspended"]
    assert result.loc["SH600000", "major_violation"]
    assert "st" in second_reasons
    assert "suspended" in second_reasons
    assert "major_violation" in second_reasons
    assert not (target / ".eligibility_attempt").exists()


def test_eligibility_rejects_symbols_that_collide_after_qlib_normalization(
    tmp_path: Path,
) -> None:
    snapshot = _write_market_control_snapshot(
        tmp_path, ts_code="000001.SZ", up_limit=11.0, down_limit=9.0
    )
    daily_path = next((snapshot / "parquet" / "daily").rglob("*.parquet"))
    daily = pd.read_parquet(daily_path)
    duplicate = daily.iloc[0].copy()
    duplicate["ts_code"] = "000001.sz"
    pd.concat([daily, duplicate.to_frame().T], ignore_index=True).to_parquet(
        daily_path, index=False
    )

    with pytest.raises(ValueError, match="collide after Qlib normalization"):
        QlibBuilder(snapshot)._write_eligibility_metadata(tmp_path / "metadata")


def test_eligibility_requires_trade_calendar_for_announcement_fallback(
    tmp_path: Path,
) -> None:
    snapshot = _write_market_control_snapshot(
        tmp_path, ts_code="000001.SZ", up_limit=11.0, down_limit=9.0
    )
    anns_root = snapshot / "parquet" / "anns_d" / "partition_year=2024"
    anns_root.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "ann_date": "2024-01-02",
                "title": "关于重大违法事项的公告",
            }
        ]
    ).to_parquet(anns_root / "announcements.parquet", index=False)

    with pytest.raises(ValueError, match="requires a valid trading calendar"):
        QlibBuilder(snapshot)._write_eligibility_metadata(tmp_path / "metadata")


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


def _write_cross_source_adjustment_snapshot(
    tmp_path: Path,
    *,
    primary_pre_close: float | None,
) -> Path:
    snapshot = tmp_path / "snapshot"
    fixtures = {
        "daily": [
            {
                "ts_code": "600000.SH",
                "trade_date": "2015-12-31",
                "open": 10.0,
                "high": 10.5,
                "low": 9.5,
                "close": 10.0,
                "pre_close": 9.8,
                "vol": 100.0,
                "amount": 100.0,
                "pct_chg": 2.04,
            },
            {
                "ts_code": "600000.SH",
                "trade_date": "2016-01-04",
                "open": 10.0,
                "high": 11.5,
                "low": 9.8,
                "close": 11.0,
                "pre_close": primary_pre_close,
                "vol": 100.0,
                "amount": 110.0,
                "pct_chg": 10.0,
            },
        ],
        # BaoStock and Tushare agree on the relative path but use different
        # positive constants (5 versus 2) on opposite sides of the boundary.
        "adj_factor": [
            {
                "ts_code": "600000.SH",
                "trade_date": "2015-12-31",
                "adj_factor": 5.0,
            },
            {
                "ts_code": "600000.SH",
                "trade_date": "2016-01-04",
                "adj_factor": 2.0,
            },
        ],
        "stk_limit": [
            {
                "ts_code": "600000.SH",
                "trade_date": "2015-12-31",
                "up_limit": 11.0,
                "down_limit": 9.0,
            },
            {
                "ts_code": "600000.SH",
                "trade_date": "2016-01-04",
                "up_limit": 12.1,
                "down_limit": 9.9,
            },
        ],
    }
    for dataset, rows in fixtures.items():
        target = snapshot / "parquet" / dataset / "data.parquet"
        target.parent.mkdir(parents=True)
        pd.DataFrame(rows).to_parquet(target, index=False)
    _write_required_research_inputs(snapshot)
    stock_basic_path = next((snapshot / "parquet" / "stock_basic").rglob("*.parquet"))
    stock_basic = pd.read_parquet(stock_basic_path)
    stock_basic["ts_code"] = "600000.SH"
    stock_basic.to_parquet(stock_basic_path, index=False)
    return snapshot


def test_rebases_legacy_adjustment_factor_without_mutating_snapshot(
    tmp_path: Path,
) -> None:
    snapshot = _write_cross_source_adjustment_snapshot(
        tmp_path,
        primary_pre_close=10.0,
    )
    source_factor_path = next((snapshot / "parquet" / "adj_factor").rglob("*.parquet"))
    source_factors = pd.read_parquet(source_factor_path)["adj_factor"].tolist()
    builder = QlibBuilder(snapshot)

    evidence = builder._adjustment_boundary_evidence()
    by_symbol = builder.build_staging(tmp_path / "staging")
    frame = pd.read_parquet(by_symbol / "SH600000.parquet")
    adjusted_close = builder._load_adjusted_close()

    assert evidence["status"] == "pass"
    assert evidence["rebased_symbol_count"] == 1
    assert evidence["masked_symbol_count"] == 0
    assert evidence["rebased_symbols"][0]["legacy_scale"] == pytest.approx(0.4)
    assert len(evidence["evidence_sha256"]) == 64
    # The old segment is rebased from factor 5 to factor 2. The normalized
    # adjusted close follows the real 10% move instead of jumping to 0.44.
    assert frame["close"].tolist() == pytest.approx([1.0, 1.1])
    assert frame["factor"].tolist() == pytest.approx([0.1, 0.1])
    assert adjusted_close is not None
    assert adjusted_close["adj_close"].tolist() == pytest.approx([20.0, 22.0])
    assert pd.read_parquet(source_factor_path)["adj_factor"].tolist() == source_factors


def test_masks_cross_source_instrument_when_boundary_price_is_not_proven(
    tmp_path: Path,
) -> None:
    snapshot = _write_cross_source_adjustment_snapshot(
        tmp_path,
        primary_pre_close=8.0,
    )
    post_only = {
        "daily": {
            "ts_code": "000001.SZ",
            "trade_date": "2016-01-04",
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.0,
            "pre_close": 10.0,
            "vol": 100.0,
            "amount": 100.0,
            "pct_chg": 0.0,
        },
        "adj_factor": {
            "ts_code": "000001.SZ",
            "trade_date": "2016-01-04",
            "adj_factor": 1.0,
        },
        "stk_limit": {
            "ts_code": "000001.SZ",
            "trade_date": "2016-01-04",
            "up_limit": 11.0,
            "down_limit": 9.0,
        },
    }
    for dataset, row in post_only.items():
        path = next((snapshot / "parquet" / dataset).rglob("*.parquet"))
        pd.concat(
            [pd.read_parquet(path), pd.DataFrame([row])],
            ignore_index=True,
        ).to_parquet(path, index=False)
    builder = QlibBuilder(snapshot)

    evidence = builder._adjustment_boundary_evidence()

    assert evidence["status"] == "failed"
    assert evidence["rebased_symbol_count"] == 0
    assert evidence["masked_symbol_count"] == 1
    assert evidence["masked_ratio"] == pytest.approx(1.0)
    assert evidence["max_masked_ratio"] == pytest.approx(0.01)
    assert evidence["masked_symbols"][0]["reason"] == "boundary_price_mismatch"
    with pytest.raises(RuntimeError, match="cross-source adjustment boundary rejected"):
        builder.build_staging(tmp_path / "staging")


def test_boundary_evidence_does_not_skip_a_missing_anchor_factor(
    tmp_path: Path,
) -> None:
    snapshot = _write_cross_source_adjustment_snapshot(
        tmp_path,
        primary_pre_close=10.0,
    )
    factor_path = next((snapshot / "parquet" / "adj_factor").rglob("*.parquet"))
    factors = pd.read_parquet(factor_path)
    factors.loc[factors["trade_date"] == "2015-12-31", "trade_date"] = "2015-12-30"
    factors.to_parquet(factor_path, index=False)

    evidence = QlibBuilder(snapshot)._adjustment_boundary_evidence()

    assert evidence["status"] == "failed"
    assert evidence["cross_source_symbols"] == 1
    assert evidence["masked_symbols"][0]["reason"] == (
        "missing_or_invalid_boundary_factor"
    )


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
        "moneyflow": [
            {
                "ts_code": "000001.SZ",
                "trade_date": row["trade_date"],
                "net_mf_amount": net_amount,
                "buy_lg_amount": 30.0,
                "sell_lg_amount": 20.0,
                "buy_elg_amount": 10.0,
                "sell_elg_amount": 10.0,
            }
            for row, net_amount in zip(rows, (5.0, -2.0, 0.0), strict=True)
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
    assert frame["mf_net_inflow_amount"].tolist() == pytest.approx(
        [50_000.0, -20_000.0, 0.0]
    )
    assert frame["mf_net_inflow_ratio"].tolist() == pytest.approx([0.5, -0.2, 0.0])
    assert frame["mf_large_order_imbalance"].tolist() == pytest.approx(
        [1.0 / 7.0] * 3
    )
    lag_label = f"effective_date_with_lag(days={METADATA_AVAILABILITY_LAG_DAYS})"
    assert builder.research_feature_contract["availability_policy"] == {
        "daily_basic": "same_trade_date_after_close",
        "moneyflow": "same_trade_date_after_close",
        "fina_indicator": "strictly_after_announcement_date",
        "fina_indicator_nondefault": "strictly_after_announcement_date",
        "income": "strictly_after_announcement_date",
        "balancesheet": "strictly_after_announcement_date",
        "cashflow": "strictly_after_announcement_date",
        "index_weight": lag_label,
        "index_member_all": lag_label,
    }
    assert builder.research_feature_contract["recoverability"] == {
        "daily_basic": "native_history",
        "moneyflow": "native_history",
        "fina_indicator": "native_history",
        "fina_indicator_nondefault": "native_history",
        "income": "native_history",
        "balancesheet": "native_history",
        "cashflow": "native_history",
        "index_weight": "native_history",
        "index_member_all": "reconstructed",
    }
    assert "fund_roe" in builder.qlib_fields
    assert "mf_net_inflow_ratio" in builder.qlib_fields
    assert builder._field_units()["mf_net_inflow_amount"] == "cny_yuan"


def test_accepts_tushare_unrestricted_price_limit_sentinel(tmp_path: Path) -> None:
    snapshot = _write_market_control_snapshot(
        tmp_path,
        ts_code="920690.BJ",
        up_limit=99999.99,
        down_limit=0.0,
    )
    stock_basic_path = next((snapshot / "parquet" / "stock_basic").rglob("*.parquet"))
    stock_basic = pd.read_parquet(stock_basic_path)
    stock_basic["ts_code"] = "920690.BJ"
    stock_basic.to_parquet(stock_basic_path, index=False)

    by_symbol = QlibBuilder(snapshot).build_staging(tmp_path / "staging")
    frame = pd.read_parquet(by_symbol / "BJ920690.parquet")

    assert frame["up_limit"].iloc[0] == pytest.approx(9999.999)
    assert frame["down_limit"].iloc[0] == 0.0


def test_missing_native_price_limits_are_research_only_unrestricted_rows(
    tmp_path: Path,
) -> None:
    snapshot = _write_market_control_snapshot(
        tmp_path,
        ts_code="000001.SZ",
        up_limit=11.0,
        down_limit=9.0,
    )
    daily_path = snapshot / "parquet" / "daily" / "partition_year=2024" / "data.parquet"
    daily = pd.read_parquet(daily_path)
    next_session = daily.iloc[0].copy()
    next_session["trade_date"] = "2024-01-08"
    pd.concat([daily, next_session.to_frame().T], ignore_index=True).to_parquet(
        daily_path, index=False
    )
    adj_path = snapshot / "parquet" / "adj_factor" / "partition_year=2024" / "data.parquet"
    adjustment = pd.read_parquet(adj_path)
    next_adjustment = adjustment.iloc[0].copy()
    next_adjustment["trade_date"] = "2024-01-08"
    pd.concat(
        [adjustment, next_adjustment.to_frame().T], ignore_index=True
    ).to_parquet(adj_path, index=False)
    limit_path = snapshot / "parquet" / "stk_limit" / "partition_year=2024" / "data.parquet"
    pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "2024-01-08",
                "up_limit": 11.0,
                "down_limit": 9.0,
            }
        ]
    ).to_parquet(limit_path)

    builder = QlibBuilder(snapshot)
    by_symbol = builder.build_staging(tmp_path / "staging")
    frame = pd.read_parquet(by_symbol / "SZ000001.parquet")
    coverage = builder._execution_control_coverage()

    assert frame["up_limit"].iloc[0] == pytest.approx(9999.999)
    assert frame["down_limit"].iloc[0] == 0.0
    assert coverage == {
        "source": "native_stk_limit",
        "scope_version": GOVERNED_DAILY_STOCK_SCOPE_VERSION,
        "scope": {
            "version": GOVERNED_DAILY_STOCK_SCOPE_VERSION,
            "security_master": "stock_basic",
            "allowed_exchanges": ["SH", "SZ", "BJ"],
            "excluded_b_share_code_patterns": ["20*.SZ", "900*.SH"],
            "invalid_code_policy": "exclude_outside_frozen_a_share_code_families",
            "bse_history_start": BSE_GOVERNED_HISTORY_START.isoformat(),
            "historical_lifecycle_inference": {
                "evidence_status": "verified",
                "qualification_sources": ["daily", "daily_basic"],
                "qualification_join": "same_ts_code_and_trade_date",
                "lifecycle_bounds": "governed_daily_min_max",
                "stock_basic_symbol_count": 1,
                "inferred_symbol_count": 0,
                "inferred_symbols_sha256": hashlib.sha256(b"[]").hexdigest(),
                "inferred_symbols": [],
            },
        },
        "missing_rows": 1,
        "total_rows": 2,
        "raw_total_rows": 2,
        "excluded_rows": 0,
        "first_missing_date": "2024-01-02",
        "last_missing_date": "2024-01-02",
        "native_complete_from": "2024-01-08",
        "missing_row_policy": "research_only_unrestricted_sentinel",
        "formal_execution_requires_native_controls": True,
    }


def test_governed_stock_scope_is_shared_by_publication_controls_and_eligibility(
    tmp_path: Path,
) -> None:
    snapshot = _write_market_control_snapshot(
        tmp_path,
        ts_code="000001.SZ",
        up_limit=11.0,
        down_limit=9.0,
    )
    daily_path = snapshot / "parquet" / "daily" / "partition_year=2024" / "data.parquet"
    base_daily = pd.read_parquet(daily_path).iloc[0].to_dict()
    extra_daily = [
        {**base_daily, "ts_code": "200001.SZ"},
        {**base_daily, "ts_code": "900901.SH"},
        {**base_daily, "ts_code": "201872.SZ"},
        {**base_daily, "ts_code": "600123.SH"},
        {**base_daily, "ts_code": "300114.SZ", "trade_date": "2018-01-02"},
        {**base_daily, "ts_code": "300114.SZ", "trade_date": "2025-02-14"},
        {**base_daily, "ts_code": "920001.BJ", "trade_date": "2022-12-30"},
        {**base_daily, "ts_code": "920001.BJ", "trade_date": "2023-01-03"},
    ]
    pd.DataFrame([base_daily, *extra_daily]).to_parquet(daily_path, index=False)

    basic_path = next((snapshot / "parquet" / "daily_basic").rglob("*.parquet"))
    daily_basic = pd.read_parquet(basic_path)
    jointly_evidenced = [
        row
        for row in extra_daily
        if row["ts_code"] != "600123.SH"
    ]
    pd.concat(
        [
            daily_basic,
            pd.DataFrame(
                [
                    {
                        "ts_code": row["ts_code"],
                        "trade_date": row["trade_date"],
                        "total_mv": 100_000.0,
                    }
                    for row in jointly_evidenced
                ]
            ),
        ],
        ignore_index=True,
    ).to_parquet(basic_path, index=False)

    adj_path = snapshot / "parquet" / "adj_factor" / "partition_year=2024" / "data.parquet"
    pd.DataFrame(
        [
            {
                "ts_code": row["ts_code"],
                "trade_date": row["trade_date"],
                "adj_factor": 1.0,
            }
            for row in [base_daily, *extra_daily]
        ]
    ).to_parquet(adj_path, index=False)
    limit_path = snapshot / "parquet" / "stk_limit" / "partition_year=2024" / "data.parquet"
    pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "2024-01-02",
                "up_limit": 11.0,
                "down_limit": 9.0,
            },
            {
                "ts_code": "920001.BJ",
                "trade_date": "2023-01-03",
                "up_limit": 11.0,
                "down_limit": 9.0,
            },
            {
                "ts_code": "300114.SZ",
                "trade_date": "2018-01-02",
                "up_limit": 11.0,
                "down_limit": 9.0,
            },
            {
                "ts_code": "300114.SZ",
                "trade_date": "2025-02-14",
                "up_limit": 11.0,
                "down_limit": 9.0,
            },
        ]
    ).to_parquet(limit_path, index=False)

    stock_basic_path = next((snapshot / "parquet" / "stock_basic").rglob("*.parquet"))
    stock_basic = pd.read_parquet(stock_basic_path)
    pd.concat(
        [
            stock_basic,
            pd.DataFrame(
                [
                    {
                        "ts_code": "200001.SZ",
                        "list_date": "2020-01-02",
                        "delist_date": None,
                    },
                    {
                        "ts_code": "900901.SH",
                        "list_date": "2020-01-02",
                        "delist_date": None,
                    },
                    {
                        "ts_code": "920001.BJ",
                        "list_date": "2022-01-02",
                        "delist_date": None,
                    },
                ]
            ),
        ],
        ignore_index=True,
    ).to_parquet(stock_basic_path, index=False)

    builder = QlibBuilder(snapshot)
    by_symbol = builder.build_staging(tmp_path / "staging")
    assert {path.stem for path in by_symbol.glob("*.parquet")} == {
        "BJ920001",
        "SZ300114",
        "SZ000001",
    }
    bj = pd.read_parquet(by_symbol / "BJ920001.parquet")
    assert pd.to_datetime(bj["date"]).dt.date.astype(str).tolist() == ["2023-01-03"]

    coverage = builder._execution_control_coverage()
    assert coverage["scope_version"] == GOVERNED_DAILY_STOCK_SCOPE_VERSION
    assert coverage["missing_rows"] == 0
    assert coverage["total_rows"] == 4
    assert coverage["raw_total_rows"] == 9
    assert coverage["excluded_rows"] == 5
    assert coverage["native_complete_from"] == "2018-01-02"
    inference = coverage["scope"]["historical_lifecycle_inference"]
    assert inference["inferred_symbol_count"] == 1
    assert inference["inferred_symbols"] == [
        {
            "ts_code": "300114.SZ",
            "first_session": "2018-01-02",
            "last_session": "2025-02-14",
            "daily_rows": 2,
            "matching_daily_basic_rows": 2,
            "source": "daily_and_daily_basic_same_session_inference",
        }
    ]

    qlib_dir = tmp_path / "qlib"
    (qlib_dir / "instruments").mkdir(parents=True)
    builder._write_stock_universe(qlib_dir)
    universe = (qlib_dir / "instruments" / "cn_all.txt").read_text(encoding="utf-8")
    assert "BJ920001\t2023-01-03\t2023-01-03" in universe
    assert "SZ300114\t2018-01-02\t2025-02-14" in universe
    assert "SZ000001\t2024-01-02\t2024-01-02" in universe
    assert all(
        value not in universe
        for value in ("SZ200001", "SH900901", "SZ201872", "SH600123")
    )

    style_metadata = tmp_path / "style-metadata"
    assert builder._write_style_metadata_bounded(style_metadata) is True
    styles = pd.read_parquet(style_metadata / "style_exposures.parquet")
    assert "SZ300114" in set(styles["instrument"])
    assert not {
        "SZ200001",
        "SH900901",
        "SZ201872",
        "SH600123",
    }.intersection(styles["instrument"])

    metadata = tmp_path / "metadata"
    assert builder._write_eligibility_metadata(metadata) is True
    eligibility = pd.read_parquet(metadata / "eligibility_matrix.parquet")
    assert set(eligibility["instrument"]) == {
        "BJ920001",
        "SZ000001",
        "SZ300114",
    }
    bj_eligibility = eligibility[eligibility["instrument"].eq("BJ920001")]
    assert pd.to_datetime(bj_eligibility["datetime"]).dt.date.min() >= BSE_GOVERNED_HISTORY_START
    eligibility_contract = json.loads(
        (metadata / "eligibility_contract.json").read_text(encoding="utf-8")
    )
    assert (
        eligibility_contract["governed_stock_scope"]["version"]
        == GOVERNED_DAILY_STOCK_SCOPE_VERSION
    )


def test_native_execution_coverage_fails_closed_without_stock_master(tmp_path: Path) -> None:
    snapshot = _write_market_control_snapshot(
        tmp_path,
        ts_code="000001.SZ",
        up_limit=11.0,
        down_limit=9.0,
        include_research_inputs=False,
    )

    with pytest.raises(ValueError, match="requires a stock_basic security master"):
        QlibBuilder(snapshot)._execution_control_coverage()


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
    assert "missing moneyflow" in message
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
    membership_path = (
        snapshot
        / "parquet"
        / "index_member_all"
        / "partition_year=2024"
        / "research.parquet"
    )
    pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "l1_code": "801780.SI",
                "in_date": "2021-01-01",
                "out_date": None,
            },
            {
                # This constituent is classified by the latest weight date,
                # but not by the earlier date. A latest-only gate would miss
                # the historical hole.
                "ts_code": "600000.SH",
                "l1_code": "801780.SI",
                "in_date": "2024-02-01",
                "out_date": None,
            },
        ]
    ).to_parquet(membership_path)
    pd.DataFrame(
        [
            {
                "index_code": "000300.SH",
                "con_code": instrument,
                "trade_date": trade_date,
                "weight": weight,
            }
            for trade_date in ("2024-01-02", "2024-03-01")
            for instrument, weight in (
                ("000001.SZ", 97.9),
                ("600000.SH", 2.1),
            )
        ]
    ).to_parquet(weight_path)

    with pytest.raises(
        RuntimeError,
        match=(
            "no active point-in-time industry for 1/4 "
            "000300.SH constituent-date rows across 2 benchmark dates.*"
            "2.1000% exceeds 2.00%"
        ),
    ):
        QlibBuilder(snapshot).build_staging(tmp_path / "staging")


def test_accepts_bounded_unknown_benchmark_industry_weight(tmp_path: Path) -> None:
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
    membership_path = (
        snapshot
        / "parquet"
        / "index_member_all"
        / "partition_year=2024"
        / "research.parquet"
    )
    pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "l1_code": "801780.SI",
                "in_date": "2021-01-01",
                "out_date": None,
            },
            {
                "ts_code": "600000.SH",
                "l1_code": "   ",
                "in_date": "2021-01-01",
                "out_date": None,
            },
        ]
    ).to_parquet(membership_path)
    pd.DataFrame(
        [
            {
                "index_code": "000300.SH",
                "con_code": instrument,
                "trade_date": trade_date,
                "weight": weight,
            }
            for trade_date in ("2024-01-02", "2024-03-01")
            for instrument, weight in (
                ("000001.SZ", 98.5),
                ("600000.SH", 1.5),
            )
        ]
    ).to_parquet(weight_path)

    builder = QlibBuilder(snapshot)
    by_symbol = builder.build_staging(tmp_path / "staging")
    qlib_dir = tmp_path / "qlib"
    builder._write_portfolio_metadata(qlib_dir)
    memberships = pd.read_parquet(
        qlib_dir / "metadata" / "industry_memberships.parquet"
    )

    assert (by_symbol / "SZ000001.parquet").exists()
    assert memberships.loc[
        memberships["instrument"] == "SH600000", "industry"
    ].tolist() == ["__UNKNOWN__"]
    assert not memberships["industry"].astype("string").str.strip().eq("").any()


def test_rejects_blank_industry_above_unknown_weight_limit(tmp_path: Path) -> None:
    snapshot = _write_market_control_snapshot(
        tmp_path,
        ts_code="000001.SZ",
        up_limit=11.0,
        down_limit=9.0,
    )
    membership_path = (
        snapshot
        / "parquet"
        / "index_member_all"
        / "partition_year=2024"
        / "research.parquet"
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
                "ts_code": "000001.SZ",
                "l1_code": "801780.SI",
                "in_date": "2021-01-01",
                "out_date": None,
            },
            {
                "ts_code": "600000.SH",
                "l1_code": "\t ",
                "in_date": "2021-01-01",
                "out_date": None,
            },
        ]
    ).to_parquet(membership_path)
    pd.DataFrame(
        [
            {
                "index_code": "000300.SH",
                "con_code": instrument,
                "trade_date": "2024-01-02",
                "weight": weight,
            }
            for instrument, weight in (
                ("000001.SZ", 97.9),
                ("600000.SH", 2.1),
            )
        ]
    ).to_parquet(weight_path)

    with pytest.raises(
        RuntimeError,
        match=(
            "no active point-in-time industry for 1/2 "
            "000300.SH constituent-date rows.*2.1000% exceeds 2.00%"
        ),
    ):
        QlibBuilder(snapshot).build_staging(tmp_path / "staging")


def test_blank_instrument_is_not_usable_for_industry(tmp_path: Path) -> None:
    snapshot = _write_market_control_snapshot(
        tmp_path,
        ts_code="000001.SZ",
        up_limit=11.0,
        down_limit=9.0,
    )
    membership_path = (
        snapshot
        / "parquet"
        / "index_member_all"
        / "partition_year=2024"
        / "research.parquet"
    )
    pd.DataFrame(
        [
            {
                "ts_code": " \t ",
                "l1_code": "801780.SI",
                "in_date": "2021-01-01",
                "out_date": None,
            }
        ]
    ).to_parquet(membership_path)

    with pytest.raises(RuntimeError, match="index_member_all has no usable rows"):
        QlibBuilder(snapshot).build_staging(tmp_path / "staging")


def test_rejects_overlapping_distinct_l1_industry_intervals(tmp_path: Path) -> None:
    snapshot = _write_market_control_snapshot(
        tmp_path,
        ts_code="000001.SZ",
        up_limit=11.0,
        down_limit=9.0,
    )
    membership_path = (
        snapshot
        / "parquet"
        / "index_member_all"
        / "partition_year=2024"
        / "research.parquet"
    )
    pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "l1_code": "801780.SI",
                "in_date": "2021-01-01",
                "out_date": "2024-01-02",
            },
            {
                "ts_code": "000001.SZ",
                "l1_code": "801010.SI",
                "in_date": "2024-01-02",
                "out_date": None,
            },
        ]
    ).to_parquet(membership_path)

    with pytest.raises(
        RuntimeError,
        match=(
            "overlapping distinct point-in-time L1 industry intervals.*"
            "2024-01-02.*000001.SZ:801010.SI/801780.SI"
        ),
    ):
        QlibBuilder(snapshot).build_staging(tmp_path / "staging")


def test_accepts_adjacent_l1_industry_interval_switch(tmp_path: Path) -> None:
    snapshot = _write_market_control_snapshot(
        tmp_path,
        ts_code="000001.SZ",
        up_limit=11.0,
        down_limit=9.0,
    )
    membership_path = (
        snapshot
        / "parquet"
        / "index_member_all"
        / "partition_year=2024"
        / "research.parquet"
    )
    pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "l1_code": "801780.SI",
                "in_date": "2021-01-01",
                "out_date": "2024-01-01",
            },
            {
                "ts_code": "000001.SZ",
                "l1_code": "801010.SI",
                "in_date": "2024-01-02",
                "out_date": None,
            },
        ]
    ).to_parquet(membership_path)

    by_symbol = QlibBuilder(snapshot).build_staging(tmp_path / "staging")

    assert (by_symbol / "SZ000001.parquet").exists()


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
                "weight": 99.5,
            },
            {
                "index_code": "000300.SH",
                "con_code": "600001.SH",
                "trade_date": "20240131",
                "weight": 0.5,
            },
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
    assert metadata["instrument"].tolist() == ["SH600000", "SH600001", "SZ000001"]
    assert metadata.loc[
        metadata["instrument"] == "SH600001", "industry"
    ].tolist() == ["__UNKNOWN__"]
    assert metadata.loc[metadata["instrument"] == "SH600000", "out_date"].notna().all()
    weights = pd.read_parquet(qlib_dir / "metadata" / "benchmark_weights.parquet")
    assert set(weights["benchmark"]) == {"SH000300"}
    assert weights.loc[weights["instrument"] == "SZ000001", "weight"].iloc[0] == pytest.approx(
        0.995
    )
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
    assert provenance["adjustment_boundary"]["status"] == "not_applicable"
    assert len(provenance["adjustment_boundary"]["evidence_sha256"]) == 64
    assert len(provenance["field_coverage_sha256"]) == 64
    assert provenance["field_year_coverage"]["evidence_status"] == (
        "missing_normalized_staging"
    )
    assert provenance["output_manifest"]["version"] == "qlib-output-files-v1"
    assert [item["path"] for item in provenance["output_manifest"]["files"]] == [
        "metadata/adjustment_boundary.json",
        "metadata/field_year_coverage.json",
        "metadata/governed_etf_whitelist.json",
        "metadata/research_feature_contract.json"
    ]
    verify_qlib_output_manifest(qlib_dir, provenance)


def test_field_year_coverage_uses_actual_rows_and_governs_legacy_transition(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    manifest_path = snapshot / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_contracts": {
                    "primary": "tushare-compatible",
                    "legacy_market": "baostock-0.9.3",
                    "legacy_overlap_policy_version": "overlap-v1",
                }
            }
        ),
        encoding="utf-8",
    )
    builder = QlibBuilder(snapshot)
    staging = tmp_path / "staging"
    staging.mkdir()
    rows = []
    for year in range(2008, 2017):
        session = f"{year}-01-04"
        row = {"date": session, "symbol": "SH600000"}
        row.update({field: 1.0 for field in builder.qlib_fields})
        rows.append(row)
    pd.DataFrame(rows).to_parquet(staging / "SH600000.parquet", index=False)

    admitted = builder._field_year_coverage(staging)

    close = admitted["fields"]["close"]
    assert close["available_from"] == "2008-01-04"
    assert close["research_available_from"] == "2008-01-04"
    assert close["years"][0]["source_contracts"] == ["baostock-0.9.3"]
    assert close["years"][-1]["source_contracts"] == ["tushare-compatible"]
    assert len(admitted["coverage_sha256"]) == 64

    manifest_path.write_text(
        json.dumps(
            {
                "source_contracts": {
                    "primary": "tushare-compatible",
                    "legacy_market": "baostock-0.9.3",
                }
            }
        ),
        encoding="utf-8",
    )
    unverified = QlibBuilder(snapshot)._field_year_coverage(staging)
    assert unverified["fields"]["close"]["available_from"] == "2008-01-04"
    assert unverified["fields"]["close"]["research_available_from"] == (
        "2016-01-01"
    )


def test_qlib_output_manifest_rejects_changed_or_unsealed_files(tmp_path: Path) -> None:
    qlib_dir = tmp_path / "qlib"
    feature = qlib_dir / "features" / "sh600000" / "close.day.bin"
    feature.parent.mkdir(parents=True)
    feature.write_bytes(b"sealed")
    provenance = {"output_manifest": build_qlib_output_manifest(qlib_dir)}

    verify_qlib_output_manifest(qlib_dir, provenance)
    feature.write_bytes(b"changed")
    with pytest.raises(ValueError, match="sealed manifest"):
        verify_qlib_output_manifest(qlib_dir, provenance)

    feature.write_bytes(b"sealed")
    (qlib_dir / "calendars").mkdir()
    (qlib_dir / "calendars" / "day.txt").write_text("2024-01-02\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sealed manifest"):
        verify_qlib_output_manifest(qlib_dir, provenance)


def test_derives_stable_qlib_lineage_only_from_verified_source_lineage(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    lineage_configuration = {"profile": "full", "start_date": "2024-01-01"}
    manifest = {
        "name": "snapshot",
        "profile": "full",
        "lineage_id": make_lineage_id("qlib_daily_source", lineage_configuration),
        "lineage_contract": {
            "kind": "qlib_daily_source",
            "configuration": lineage_configuration,
        },
        "lineage_generation": 0,
        "parent_snapshot": None,
        "parent_manifest_sha256": None,
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
    assert provenance["source_lineage_generation"] == 0


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


def test_bounded_daily_unit_outliers_are_excluded_without_fabricating_vwap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _write_market_control_snapshot(
        tmp_path,
        ts_code="000001.SZ",
        up_limit=11.0,
        down_limit=9.0,
    )
    daily = next((snapshot / "parquet" / "daily").rglob("*.parquet"))
    invalid = pd.read_parquet(daily).assign(ts_code="000002.SZ", amount=1_000.0)
    pd.concat([pd.read_parquet(daily), invalid], ignore_index=True).to_parquet(daily, index=False)
    for dataset in ("adj_factor", "stk_limit"):
        path = next((snapshot / "parquet" / dataset).rglob("*.parquet"))
        original = pd.read_parquet(path)
        pd.concat(
            [original, original.assign(ts_code="000002.SZ")], ignore_index=True
        ).to_parquet(path, index=False)
    builder = QlibBuilder(snapshot)
    monkeypatch.setattr(
        builder,
        "_daily_unit_quality_coverage",
        lambda: {
            "policy": "exclude_internally_inconsistent_rows_from_research_history",
            "excluded_rows": 1,
            "total_rows": 100_001,
            "excluded_ratio": 1 / 100_001,
            "max_excluded_ratio": _MAX_EXCLUDED_DAILY_UNIT_RATIO,
        },
    )

    by_symbol = builder.build_staging(tmp_path / "staging")

    assert (by_symbol / "SZ000001.parquet").exists()
    assert not (by_symbol / "SZ000002.parquet").exists()


def test_fundamental_partitions_allow_optional_ingestion_metadata(tmp_path: Path) -> None:
    snapshot = _write_market_control_snapshot(
        tmp_path,
        ts_code="000001.SZ",
        up_limit=11.0,
        down_limit=9.0,
    )
    cashflow = snapshot / "parquet" / "cashflow"
    first = cashflow / "partition_year=2023"
    second = cashflow / "partition_year=2024"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    common = {
        "ts_code": "000001.SZ",
        "end_date": "2023-09-30",
        "n_cashflow_act": 100.0,
        "c_pay_acq_const_fiolta": 20.0,
    }
    pd.DataFrame(
        [{**common, "ann_date": "2023-12-31", "ingested_at": "2024-01-01T00:00:00Z"}]
    ).to_parquet(first / "data.parquet")
    pd.DataFrame([{**common, "ann_date": "2024-01-01"}]).to_parquet(
        second / "data.parquet"
    )

    by_symbol = QlibBuilder(snapshot).build_staging(tmp_path / "staging")
    frame = pd.read_parquet(by_symbol / "SZ000001.parquet")

    assert "fund_ocf_net" in frame.columns


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
    companion_fields = {
        "q_profit_yoy",
        "inv_turn",
        "ocf_to_or",
        "ocf_to_profit",
        "salescash_to_or",
    }
    companion_keys = {
        "ts_code",
        "ann_date",
        "end_date",
        "f_ann_date",
        "update_flag",
        "ingested_at",
    }
    default_fina_rows = [
        {key: value for key, value in row.items() if key not in companion_fields}
        for row in fina_rows
    ]
    companion_rows = [
        {
            key: value
            for key, value in row.items()
            if key in companion_keys or key in companion_fields
        }
        for row in fina_rows
        if companion_fields.intersection(row)
    ]
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
        "fina_indicator": default_fina_rows,
    }
    if companion_rows:
        fixtures["fina_indicator_nondefault"] = companion_rows
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


def test_appending_future_market_and_restatement_rows_preserves_historical_features(
    tmp_path: Path,
) -> None:
    """A later snapshot extension must be a strict prefix extension for old dates.

    This exercises the actual Qlib staging join, including adjusted market data,
    price limits, daily descriptors, and announcement-date financial revisions.
    It is intentionally stronger than checking one fundamental column: every
    emitted historical feature must remain byte-for-value equivalent after the
    source snapshot gains a later session and a later-announced restatement.
    """

    snapshot = tmp_path / "snapshot"
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
        daily_days=("2024-01-02", "2024-01-03", "2024-01-04"),
    )
    before_path = QlibBuilder(snapshot).build_staging(tmp_path / "staging-before")
    before = pd.read_parquet(before_path / "SZ000001.parquet")

    future_market_rows = {
        "daily": {
            "ts_code": "000001.SZ",
            "trade_date": "2024-01-08",
            "open": 50.0,
            "high": 55.0,
            "low": 45.0,
            "close": 50.0,
            "vol": 100.0,
            "amount": 500.0,
            "pct_chg": 400.0,
        },
        "adj_factor": {
            "ts_code": "000001.SZ",
            "trade_date": "2024-01-08",
            "adj_factor": 1.0,
        },
        "stk_limit": {
            "ts_code": "000001.SZ",
            "trade_date": "2024-01-08",
            "up_limit": 55.0,
            "down_limit": 45.0,
        },
        "daily_basic": {
            "ts_code": "000001.SZ",
            "trade_date": "2024-01-08",
            "total_mv": 100_000.0,
        },
    }
    for dataset, row in future_market_rows.items():
        target = snapshot / "parquet" / dataset / "partition_year=2024"
        pd.DataFrame([row]).to_parquet(target / "future.parquet", index=False)
    pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "ann_date": "2024-01-05",
                "end_date": "2023-12-31",
                "roe": 99.0,
            }
        ]
    ).to_parquet(
        snapshot
        / "parquet"
        / "fina_indicator"
        / "partition_year=2024"
        / "future.parquet",
        index=False,
    )

    after_path = QlibBuilder(snapshot).build_staging(tmp_path / "staging-after")
    after = pd.read_parquet(after_path / "SZ000001.parquet")
    after_dates = pd.to_datetime(after["date"])
    historical_after = after.loc[after_dates.le(pd.Timestamp("2024-01-04"))]

    pd.testing.assert_frame_equal(
        before.reset_index(drop=True),
        historical_after.reset_index(drop=True),
        check_exact=True,
    )
    before_factor = before["close"].pct_change(fill_method=None).rolling(2).mean()
    after_factor = after["close"].pct_change(fill_method=None).rolling(2).mean()
    pd.testing.assert_series_equal(
        before_factor.reset_index(drop=True),
        after_factor.iloc[: len(before_factor)].reset_index(drop=True),
        check_exact=True,
    )
    assert after_dates.max() == pd.Timestamp("2024-01-08")
    assert after.loc[after_dates.eq(pd.Timestamp("2024-01-08")), "close"].item() != before[
        "close"
    ].iloc[-1]
    assert after.loc[after_dates.eq(pd.Timestamp("2024-01-08")), "fund_roe"].item() == 99.0


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
    "q_profit_yoy": 22.0,
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
    "fund_quarter_profit_yoy": 22.0,
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
    assert contract["version"] == 6
    # A Tushare non-default column omitted by the companion response is
    # reported as missing instead of being silently skipped.
    missing = contract["missing_fundamental_fields"]["fina_indicator_nondefault"]
    assert missing["q_profit_yoy"] == "fund_quarter_profit_yoy"
    assert "ocf_to_or" not in missing
    # A column that exists but holds no non-null value is a distinct class.
    assert contract["all_null_fundamental_fields"]["fina_indicator_nondefault"] == {
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
