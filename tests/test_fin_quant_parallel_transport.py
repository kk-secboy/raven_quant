from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.no_database


@pytest.mark.skipif(sys.platform != "linux", reason="Production Pandarallel uses Linux fork")
def test_pipe_factor_deduplication_matches_upstream_without_temporary_files(monkeypatch):
    pytest.importorskip("rdagent")
    core = pytest.importorskip("pandarallel.core")
    from pandarallel import pandarallel

    from scripts.run_rdagent_module import _enable_fin_quant_pipe_transport

    initialize = pandarallel.initialize
    monkeypatch.setattr(pandarallel, "initialize",
                        lambda **kw: initialize(**{**kw, "nb_workers": 2}))
    from rdagent.scenarios.qlib.developer.factor_runner import QlibFactorRunner

    _enable_fin_quant_pipe_transport()

    def reject_tempfile(*args, **kwargs):
        raise AssertionError("factor deduplication must not allocate temporary pickle files")

    monkeypatch.setattr(core, "NamedTemporaryFile", reject_tempfile)
    rows = 2000
    index = pd.MultiIndex.from_product(
        [pd.date_range("2020-01-01", periods=10), range(rows // 10)],
        names=["datetime", "instrument"])
    rng = np.random.default_rng(42)
    sota = pd.DataFrame(rng.normal(size=(rows, 2)), index=index, columns=["a", "b"])
    proposed = pd.DataFrame({"duplicate": sota["a"], "negative": -sota["a"],
                            "independent": rng.normal(size=rows)}, index=index)
    proposed.iloc[::13, 2] = np.nan
    runner = object.__new__(QlibFactorRunner)
    actual = runner.deduplicate_new_factors(sota, proposed)
    correlations = pd.concat([sota, proposed], axis=1).groupby("datetime").apply(
        lambda group: runner.calculate_information_coefficient(group, 2, 3)).mean()
    correlations.index = pd.MultiIndex.from_product([range(2), range(3)])
    expected = proposed.iloc[:, correlations.unstack().max(axis=0).loc[lambda x: x < .99].index]
    pd.testing.assert_frame_equal(actual, expected)
    assert list(actual) == ["negative", "independent"]
