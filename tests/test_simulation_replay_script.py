from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.no_database


def _script_module():
    path = Path(__file__).parents[1] / "scripts" / "run_simulation_replay.py"
    spec = importlib.util.spec_from_file_location("run_simulation_replay_script", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_daily_settlement_uses_sealed_known_calendar_beyond_market_bars(
    tmp_path: Path,
) -> None:
    script = _script_module()
    provider = tmp_path / "qlib"
    (provider / "metadata").mkdir(parents=True)
    (provider / "calendars").mkdir()
    # Market-bar calendar is deliberately sealed only through execution day.
    (provider / "calendars" / "day.txt").write_text(
        "2026-08-27\n2026-08-28\n", encoding="utf-8"
    )
    pd.DataFrame(
        {"date": pd.to_datetime(["2026-08-27", "2026-08-28", "2026-08-31"])}
    ).to_parquet(
        provider / "metadata" / "known_trading_calendar.parquet",
        index=False,
    )
    calendar_path = provider / "metadata" / "known_trading_calendar.parquet"

    next_date, evidence = script._next_settlement_session(
        provider,
        trade_date="2026-08-28",
        dataset_identity_sha256="a" * 64,
        dataset_lineage_id="b" * 64,
        expected_calendar_file_sha256=script._sha256_file(calendar_path),
        expected_calendar_file_bytes=calendar_path.stat().st_size,
        expected_next_trade_date="2026-08-31",
    )

    assert next_date == "2026-08-31"
    assert evidence["trade_date"] == "2026-08-28"
    assert evidence["next_trade_date"] == "2026-08-31"
    assert evidence["dataset_identity_sha256"] == "a" * 64
    assert evidence["dataset_lineage_id"] == "b" * 64
    assert evidence["calendar_file_sha256"] == script._sha256_file(
        provider / "metadata" / "known_trading_calendar.parquet"
    )
    with pytest.raises(ValueError, match="hash differs from batch binding"):
        script._next_settlement_session(
            provider,
            trade_date="2026-08-28",
            dataset_identity_sha256="a" * 64,
            dataset_lineage_id="b" * 64,
            expected_calendar_file_sha256="f" * 64,
            expected_calendar_file_bytes=calendar_path.stat().st_size,
            expected_next_trade_date="2026-08-31",
        )
    with pytest.raises(ValueError, match="next session differs from batch binding"):
        script._next_settlement_session(
            provider,
            trade_date="2026-08-28",
            dataset_identity_sha256="a" * 64,
            dataset_lineage_id="b" * 64,
            expected_calendar_file_sha256=script._sha256_file(calendar_path),
            expected_calendar_file_bytes=calendar_path.stat().st_size,
            expected_next_trade_date="2026-09-01",
        )


def test_daily_settlement_calendar_fails_closed_when_execution_day_is_absent(
    tmp_path: Path,
) -> None:
    script = _script_module()
    provider = tmp_path / "qlib"
    (provider / "metadata").mkdir(parents=True)
    pd.DataFrame(
        {"date": pd.to_datetime(["2026-08-27", "2026-08-31"])}
    ).to_parquet(
        provider / "metadata" / "known_trading_calendar.parquet",
        index=False,
    )
    calendar_path = provider / "metadata" / "known_trading_calendar.parquet"

    with pytest.raises(ValueError, match="does not contain the execution session"):
        script._next_settlement_session(
            provider,
            trade_date="2026-08-28",
            dataset_identity_sha256="a" * 64,
            dataset_lineage_id="b" * 64,
            expected_calendar_file_sha256=script._sha256_file(calendar_path),
            expected_calendar_file_bytes=calendar_path.stat().st_size,
            expected_next_trade_date="2026-08-31",
        )


def test_daily_settlement_calendar_fails_closed_if_file_changes_while_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _script_module()
    provider = tmp_path / "qlib"
    (provider / "metadata").mkdir(parents=True)
    pd.DataFrame(
        {"date": pd.to_datetime(["2026-08-28", "2026-08-31"])}
    ).to_parquet(
        provider / "metadata" / "known_trading_calendar.parquet",
        index=False,
    )
    calendar_path = provider / "metadata" / "known_trading_calendar.parquet"
    observed_hashes = iter(["a" * 64, "b" * 64])
    monkeypatch.setattr(script, "_sha256_file", lambda _path: next(observed_hashes))

    with pytest.raises(ValueError, match="changed while being read"):
        script._next_settlement_session(
            provider,
            trade_date="2026-08-28",
            dataset_identity_sha256="c" * 64,
            dataset_lineage_id="d" * 64,
            expected_calendar_file_sha256="a" * 64,
            expected_calendar_file_bytes=calendar_path.stat().st_size,
            expected_next_trade_date="2026-08-31",
        )


def test_empty_execution_bars_keep_the_engine_contract() -> None:
    script = _script_module()

    values = script._empty_execution_bars()

    assert values.empty
    assert set(values.columns) == {
        "datetime",
        "instrument",
        "close",
        "vwap",
        "volume",
        "paused",
        "up_limit",
        "down_limit",
    }


def test_pair_replay_loads_only_dated_tushare_shortability_rows(
    tmp_path: Path,
) -> None:
    script = _script_module()
    source = tmp_path / "margin_eligibility"
    source.mkdir()
    pd.DataFrame(
        {
            "ts_code": [
                "600000.SH",
                "600001.SH",
                "600000.SH",
                "000001.SZ",
            ],
            "trade_date": [20260713, 20260713, 20260710, 20260713],
            "is_shortable": [1, 0, 0, 1],
        }
    ).to_parquet(source / "evidence.parquet", index=False)

    result = script._load_shortability(
        source,
        instruments=["SH600000", "SH600001"],
        trade_date="2026-07-13",
    )

    assert result == {"SH600000": True, "SH600001": False}


def test_pair_replay_blocks_when_tushare_has_no_dated_leg_evidence(
    tmp_path: Path,
) -> None:
    script = _script_module()
    source = tmp_path / "margin_eligibility"
    source.mkdir()
    pd.DataFrame(
        {
            "ts_code": ["600000.SH"],
            "trade_date": [20260713],
            "is_shortable": [1],
        }
    ).to_parquet(source / "evidence.parquet", index=False)

    with pytest.raises(ValueError, match="no dated evidence"):
        script._load_shortability(
            source,
            instruments=["SH600000", "SH600001"],
            trade_date="2026-07-13",
        )


def test_vwap_profile_uses_only_prior_bound_qlib_volume_rows() -> None:
    script = _script_module()

    class DataApi:
        @staticmethod
        def calendar(**_kwargs):
            return pd.to_datetime(
                [
                    "2026-07-09 10:00:00",
                    "2026-07-10 10:00:00",
                    "2026-07-13 10:00:00",
                ]
            )

        @staticmethod
        def features(*_args, **_kwargs):
            return pd.DataFrame(
                {
                    "datetime": pd.to_datetime(
                        [
                            "2026-07-09 10:00:00",
                            "2026-07-09 14:50:00",
                            "2026-07-10 10:00:00",
                            "2026-07-10 14:50:00",
                            "2026-07-13 10:00:00",
                        ]
                    ),
                    "instrument": ["SH600000"] * 5,
                    "$volume": [100.0, 300.0, 200.0, 500.0, 999_999.0],
                }
            ).set_index(["datetime", "instrument"])

    profile, evidence, digest = script._historical_volume_profile(
        DataApi,
        instruments=["SH600000"],
        trade_date="2026-07-13",
        frequency="5min",
        execution_policy={
            "execution_algorithm": "vwap",
            "slice_minutes": 20,
            "max_slices": 2,
            "max_participation": 0.01,
            "volume_profile_method": script.VWAP_PROFILE_METHOD,
            "volume_profile_lookback_days": 2,
            "simulation_semantics_sha256": "a" * 64,
        },
        dataset_identity_sha256="b" * 64,
        dataset_lineage_id="c" * 64,
    )

    assert profile == [
        {"time": "10:00", "weight": 150.0},
        {"time": "14:50", "weight": 400.0},
    ]
    assert evidence["start"] == "2026-07-09"
    assert evidence["end"] == "2026-07-10"
    assert evidence["future_data_used"] is False
    assert len(digest) == 64


def test_benchmark_evidence_uses_exact_signal_and_trade_date_closes() -> None:
    script = _script_module()

    class DataApi:
        @staticmethod
        def features(*_args, **_kwargs):
            return pd.DataFrame(
                {
                    "datetime": pd.to_datetime(
                        [
                            "2026-07-10 14:55:00",
                            "2026-07-10 15:00:00",
                            "2026-07-13 15:00:00",
                        ]
                    ),
                    "instrument": ["SH000300"] * 3,
                    "$close": [4_000.0, 4_010.0, 4_050.0],
                }
            ).set_index(["datetime", "instrument"])

    evidence = script._benchmark_evidence(
        DataApi,
        benchmark="SH000300",
        signal_date="2026-07-10",
        trade_date="2026-07-13",
        frequency="5min",
        dataset_identity_sha256="a" * 64,
        dataset_lineage_id="b" * 64,
    )

    assert evidence["baseline_close"] == pytest.approx(4_010.0)
    assert evidence["close"] == pytest.approx(4_050.0)
    assert evidence["baseline_date"] == "2026-07-10"
    assert evidence["trade_date"] == "2026-07-13"
    assert len(evidence["evidence_sha256"]) == 64
