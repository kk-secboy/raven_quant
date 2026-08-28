from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant_platform.factor_recompute import (
    FACTOR_SUBMITTED_INDEX_CONTRACT_VERSION,
    compare_submitted_values,
    execute_factor_code,
    require_exact_oos_coverage,
    submitted_comparison_is_admissible,
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


def test_factor_code_allows_local_path_boilerplate_in_the_isolated_sandbox() -> None:
    validate_factor_code(
        "import os\n"
        "from pathlib import Path\n"
        "input_path = os.path.join(os.path.dirname(__file__), 'daily_pv.h5')\n"
        "output_path = Path(__file__).parent / 'result.h5'\n"
    )


@pytest.mark.parametrize(
    "source",
    [
        "from pathlib import Path\nvalue = Path('/etc/passwd').read_text()\n",
        "import os\nvalue = list(os.walk('/'))\n",
    ],
)
def test_factor_code_rejects_path_enumeration_and_reads(source: str) -> None:
    with pytest.raises(ValueError, match="forbidden capability"):
        validate_factor_code(source)


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


def test_submitted_values_reject_coordinates_missing_from_recomputation(
    tmp_path: Path,
) -> None:
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
    assert comparison["index_subset_match"] is False
    assert comparison["overlap_rows"] == 1
    assert comparison["exact_match"] is False


def test_submitted_values_accept_a_complete_bounded_suffix(tmp_path: Path) -> None:
    pytest.importorskip("tables")
    recomputed_index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-06-30"), "SH600000"),
            (pd.Timestamp("2026-07-01"), "SH600000"),
            (pd.Timestamp("2026-07-01"), "SH600001"),
        ],
        names=["datetime", "instrument"],
    )
    submitted_index = recomputed_index[1:]
    submitted = pd.DataFrame(
        {"factor": [1.0, float("nan")]}, index=submitted_index
    )
    recomputed = pd.DataFrame(
        {"factor": [9.0, 1.0, float("nan")]}, index=recomputed_index
    )
    submitted_path = tmp_path / "submitted-subset.h5"
    submitted.to_hdf(submitted_path, key="data", mode="w")

    comparison = compare_submitted_values(submitted_path, recomputed)

    assert comparison["index_exact_match"] is False
    assert comparison["index_subset_match"] is True
    assert comparison["overlap_rows"] == 2
    assert comparison["contract_version"] == FACTOR_SUBMITTED_INDEX_CONTRACT_VERSION
    assert comparison["index_prefix_extension_match"] is True
    assert comparison["index_difference_kind"] == "recomputed_history_prefix"
    assert comparison["recomputed_history_prefix_rows"] == 1
    assert comparison["missing_on_or_after_submitted_start_rows"] == 0
    assert comparison["exact_match"] is True
    assert submitted_comparison_is_admissible(comparison) is True
    legacy_ambiguous = dict(comparison)
    legacy_ambiguous.pop("contract_version")
    assert submitted_comparison_is_admissible(legacy_ambiguous) is False
    forged_missing_tail = dict(comparison)
    forged_missing_tail["submitted_end"] = "2026-06-30T00:00:00"
    assert submitted_comparison_is_admissible(forged_missing_tail) is False


@pytest.mark.parametrize("missing_kind", ["interior", "tail"])
def test_submitted_values_reject_gapped_or_truncated_subsets(
    tmp_path: Path, missing_kind: str
) -> None:
    pytest.importorskip("tables")
    dates = pd.date_range("2026-07-01", periods=4, freq="B")
    recomputed_index = pd.MultiIndex.from_product(
        [dates, ["SH600000", "SH600001"]], names=["datetime", "instrument"]
    )
    recomputed = pd.DataFrame(
        {"factor": range(1, len(recomputed_index) + 1)},
        index=recomputed_index,
        dtype=float,
    )
    if missing_kind == "interior":
        submitted = recomputed.drop(index=(dates[2], "SH600001"))
    else:
        submitted = recomputed.loc[
            recomputed.index.get_level_values("datetime") < dates[-1]
        ]
    submitted_path = tmp_path / f"submitted-{missing_kind}.h5"
    submitted.to_hdf(submitted_path, key="data", mode="w")

    comparison = compare_submitted_values(submitted_path, recomputed)

    assert comparison["index_subset_match"] is True
    assert comparison["index_prefix_extension_match"] is False
    assert comparison["missing_on_or_after_submitted_start_rows"] > 0
    assert comparison["exact_match"] is False
    assert submitted_comparison_is_admissible(comparison) is False


def test_submitted_subset_still_requires_exact_values(tmp_path: Path) -> None:
    pytest.importorskip("tables")
    recomputed_index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-06-30"), "SH600000"),
            (pd.Timestamp("2026-07-01"), "SH600000"),
        ],
        names=["datetime", "instrument"],
    )
    submitted = pd.DataFrame(
        {"factor": [2.0]}, index=recomputed_index[1:]
    )
    recomputed = pd.DataFrame(
        {"factor": [9.0, 1.0]}, index=recomputed_index
    )
    submitted_path = tmp_path / "submitted-wrong-value.h5"
    submitted.to_hdf(submitted_path, key="data", mode="w")

    comparison = compare_submitted_values(submitted_path, recomputed)

    assert comparison["index_subset_match"] is True
    assert comparison["exact_match"] is False


def test_submitted_values_accept_only_leading_warmup_nans(tmp_path: Path) -> None:
    pytest.importorskip("tables")
    dates = pd.date_range("2026-07-01", periods=4, freq="B")
    index = pd.MultiIndex.from_product(
        [dates, ["SH600000", "SH600001"]], names=["datetime", "instrument"]
    )
    recomputed = pd.DataFrame({"factor": range(1, len(index) + 1)}, index=index, dtype=float)
    submitted = recomputed.copy()
    submitted.loc[(dates[:2], "SH600000"), "factor"] = float("nan")
    submitted.loc[(dates[:1], "SH600001"), "factor"] = float("nan")
    submitted_path = tmp_path / "submitted-warmup.h5"
    submitted.to_hdf(submitted_path, key="data", mode="w")

    comparison = compare_submitted_values(submitted_path, recomputed)

    assert comparison["exact_match"] is True
    assert comparison["finite_value_match"] is True
    assert comparison["warmup_prefix_only"] is True
    assert comparison["warmup_prefix_rows"] == 3


def test_submitted_values_reject_interior_nan_mismatches(tmp_path: Path) -> None:
    pytest.importorskip("tables")
    dates = pd.date_range("2026-07-01", periods=4, freq="B")
    index = pd.MultiIndex.from_product(
        [dates, ["SH600000"]], names=["datetime", "instrument"]
    )
    recomputed = pd.DataFrame({"factor": [1.0, 2.0, 3.0, 4.0]}, index=index)
    submitted = recomputed.copy()
    submitted.iloc[2, 0] = float("nan")
    submitted_path = tmp_path / "submitted-interior-gap.h5"
    submitted.to_hdf(submitted_path, key="data", mode="w")

    comparison = compare_submitted_values(submitted_path, recomputed)

    assert comparison["finite_value_match"] is True
    assert comparison["warmup_prefix_only"] is False
    assert comparison["exact_match"] is False


def test_submitted_values_reject_all_nan_instrument(tmp_path: Path) -> None:
    pytest.importorskip("tables")
    dates = pd.date_range("2026-07-01", periods=3, freq="B")
    index = pd.MultiIndex.from_product(
        [dates, ["SH600000", "SH600001"]], names=["datetime", "instrument"]
    )
    recomputed = pd.DataFrame({"factor": range(1, len(index) + 1)}, index=index, dtype=float)
    submitted = recomputed.copy()
    submitted.loc[(slice(None), "SH600001"), "factor"] = float("nan")
    submitted_path = tmp_path / "submitted-hidden-instrument.h5"
    submitted.to_hdf(submitted_path, key="data", mode="w")

    comparison = compare_submitted_values(submitted_path, recomputed)

    assert comparison["warmup_prefix_only"] is False
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
