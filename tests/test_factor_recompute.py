from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant_platform.factor_recompute import (
    compare_submitted_values,
    execute_factor_code,
    validate_factor_code,
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
