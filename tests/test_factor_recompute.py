from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant_platform.factor_recompute import (
    compare_submitted_values,
    execute_factor_code,
    require_exact_oos_coverage,
    validate_factor_code,
    validate_factor_prefix_invariance,
)

pytestmark = pytest.mark.no_database


def test_factor_code_is_reexecuted_against_supplied_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("tables")
    monkeypatch.setenv("FACTOR_RECOMPUTE_ALLOW_LOCAL_UNSAFE", "1")
    index = pd.MultiIndex.from_product(
        [[pd.Timestamp("2026-07-01")], ["SH600000", "SH600001"]],
        names=["datetime", "instrument"],
    )
    source = pd.DataFrame({"$close": [10.0, 20.0]}, index=index)
    input_path = tmp_path / "source.h5"
    source.to_hdf(input_path, key="data", mode="w")
    code_path = tmp_path / "candidate.py"
    code_path.write_text(
        "import pandas as pd\n"
        "frame = pd.read_hdf('daily_pv.h5', key='data')\n"
        "result = (frame['$close'] * 2).to_frame('factor')\n"
        "result.to_hdf('result.h5', key='data', mode='w')\n",
        encoding="utf-8",
    )
    values, evidence = execute_factor_code(
        code_path=code_path,
        input_path=input_path,
        workspace=tmp_path / "isolated",
    )
    assert values.iloc[:, 0].tolist() == [20.0, 40.0]
    assert evidence["code_sha256"]
    assert evidence["input_sha256"]
    assert evidence["output_sha256"]
    assert evidence["sandbox_mode"] == "local-test-override"


def test_factor_code_rejects_filesystem_and_process_capabilities() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        validate_factor_code("import os\nos.system('whoami')\n")


@pytest.mark.parametrize(
    "source",
    [
        "result = frame['$close'].shift(-1)\n",
        "result = frame['$close'].diff(periods=-2)\n",
        "result = frame['$close'].rolling(5, center=True).mean()\n",
        "result = frame['$close'].bfill()\n",
        "result = frame['$close'].fillna(method='backfill')\n",
        "result = frame['$close'].interpolate(limit_direction='both')\n",
        "result = np.roll(frame['$close'], -1)\n",
    ],
)
def test_factor_code_rejects_obvious_future_operations(source: str) -> None:
    with pytest.raises(ValueError, match="future|forward|negative|backward"):
        validate_factor_code(source)


def test_factor_recompute_fails_closed_without_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FACTOR_SANDBOX_IMAGE", raising=False)
    monkeypatch.delenv("FACTOR_RECOMPUTE_ALLOW_LOCAL_UNSAFE", raising=False)
    code = tmp_path / "factor.py"
    source = tmp_path / "input.h5"
    code.write_text("result = 1\n", encoding="utf-8")
    source.write_bytes(b"fixture")

    with pytest.raises(ValueError, match="isolated container sandbox"):
        execute_factor_code(
            code_path=code,
            input_path=source,
            workspace=tmp_path / "sandbox",
        )


def test_submitted_values_require_the_exact_recomputed_index(tmp_path: Path) -> None:
    pytest.importorskip("tables")
    submitted_index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-07-01"), "SH600000"),
            (pd.Timestamp("2026-07-01"), "SH600001"),
        ],
        names=["datetime", "instrument"],
    )
    recomputed_index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-07-01"), "SH600000"),
            (pd.Timestamp("2026-07-01"), "SH600002"),
        ],
        names=["datetime", "instrument"],
    )
    submitted = pd.DataFrame({"factor": [1.0, float("nan")]}, index=submitted_index)
    recomputed = pd.DataFrame({"factor": [1.0, float("nan")]}, index=recomputed_index)
    submitted_path = tmp_path / "submitted.h5"
    submitted.to_hdf(submitted_path, key="data", mode="w")

    comparison = compare_submitted_values(submitted_path, recomputed)

    assert comparison["index_exact_match"] is False
    assert comparison["exact_match"] is False


def _pit_input(tmp_path: Path) -> Path:
    dates = pd.date_range("2026-07-01", periods=8, freq="B")
    index = pd.MultiIndex.from_product(
        [dates, ["SH600000", "SH600001"]], names=["datetime", "instrument"]
    )
    frame = pd.DataFrame({"$close": range(1, len(index) + 1)}, index=index, dtype=float)
    input_path = tmp_path / "pit-input.h5"
    frame.to_hdf(input_path, key="data", mode="w")
    return input_path


def test_prefix_invariance_accepts_a_causal_rolling_factor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("tables")
    monkeypatch.setenv("FACTOR_RECOMPUTE_ALLOW_LOCAL_UNSAFE", "1")
    input_path = _pit_input(tmp_path)
    code_path = tmp_path / "causal.py"
    code_path.write_text(
        "import pandas as pd\n"
        "frame = pd.read_hdf('daily_pv.h5', key='data')\n"
        "result = frame['$close'].groupby(level='instrument').transform(\n"
        "    lambda value: value.rolling(2, min_periods=1).mean()\n"
        ").to_frame('factor')\n"
        "result.to_hdf('result.h5', key='data', mode='w')\n",
        encoding="utf-8",
    )
    values, _ = execute_factor_code(
        code_path=code_path,
        input_path=input_path,
        workspace=tmp_path / "causal-full",
    )

    evidence = validate_factor_prefix_invariance(
        code_path=code_path,
        input_path=input_path,
        full_values=values,
        workspace_root=tmp_path / "causal-prefixes",
    )

    assert evidence["status"] == "passed"
    assert evidence["cutpoint_count"] == 3


def test_prefix_invariance_rejects_full_sample_statistics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("tables")
    monkeypatch.setenv("FACTOR_RECOMPUTE_ALLOW_LOCAL_UNSAFE", "1")
    input_path = _pit_input(tmp_path)
    code_path = tmp_path / "leaky.py"
    code_path.write_text(
        "import pandas as pd\n"
        "frame = pd.read_hdf('daily_pv.h5', key='data')\n"
        "result = (frame['$close'] - frame['$close'].mean()).to_frame('factor')\n"
        "result.to_hdf('result.h5', key='data', mode='w')\n",
        encoding="utf-8",
    )
    values, _ = execute_factor_code(
        code_path=code_path,
        input_path=input_path,
        workspace=tmp_path / "leaky-full",
    )

    with pytest.raises(ValueError, match="prefix invariance"):
        validate_factor_prefix_invariance(
            code_path=code_path,
            input_path=input_path,
            full_values=values,
            workspace_root=tmp_path / "leaky-prefixes",
        )


def test_final_oos_coverage_rejects_silent_row_truncation(tmp_path: Path) -> None:
    pytest.importorskip("tables")
    input_path = _pit_input(tmp_path)
    factor_input = pd.read_hdf(input_path)
    truncated = factor_input[["$close"]].rename(columns={"$close": "factor"}).iloc[:-1]

    with pytest.raises(ValueError, match="does not exactly match"):
        require_exact_oos_coverage(
            truncated,
            factor_input,
            test_start="2026-07-01",
            test_end="2026-07-10",
            trading_days=pd.date_range("2026-07-01", periods=8, freq="B"),
            context="fixture",
        )


def test_final_oos_coverage_rejects_nan_filled_cross_sections(tmp_path: Path) -> None:
    pytest.importorskip("tables")
    dates = pd.date_range("2026-07-01", periods=8, freq="B")
    index = pd.MultiIndex.from_product(
        [dates, [f"SH60000{number}" for number in range(5)]],
        names=["datetime", "instrument"],
    )
    factor_input = pd.DataFrame({"$close": 1.0}, index=index)
    values = factor_input[["$close"]].rename(columns={"$close": "factor"})
    values.iloc[:, 0] = float("nan")
    first_rows = values.groupby(level="datetime").head(1).index
    values.loc[first_rows, "factor"] = 1.0

    with pytest.raises(ValueError, match="cross-sectional coverage failed"):
        require_exact_oos_coverage(
            values,
            factor_input,
            test_start="2026-07-01",
            test_end="2026-07-10",
            trading_days=dates,
            context="fixture",
            min_daily_finite=5,
        )


def test_final_oos_coverage_records_the_research_threshold_contract(tmp_path: Path) -> None:
    pytest.importorskip("tables")
    dates = pd.date_range("2026-07-01", periods=8, freq="B")
    index = pd.MultiIndex.from_product(
        [dates, [f"SH60000{number}" for number in range(5)]],
        names=["datetime", "instrument"],
    )
    factor_input = pd.DataFrame({"$close": 1.0}, index=index)
    values = factor_input[["$close"]].rename(columns={"$close": "factor"})

    evidence = require_exact_oos_coverage(
        values,
        factor_input,
        test_start="2026-07-01",
        test_end="2026-07-10",
        trading_days=dates,
        context="fixture",
        min_daily_finite=5,
    )

    assert evidence["coverage_gate_passed"] is True
    assert evidence["min_daily_finite_required"] == 5
    assert evidence["min_coverage_ratio_required"] == 0.8
    assert evidence["min_good_day_rate_required"] == 0.95
    assert evidence["good_day_rate"] == 1.0
