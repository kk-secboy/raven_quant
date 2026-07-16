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
