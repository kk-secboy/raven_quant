from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from quant_platform.simulation_store import _benchmark_chain_day

pytestmark = pytest.mark.no_database


def test_pair_cash_baseline_is_explicit_and_does_not_query_market_evidence() -> None:
    result = _benchmark_chain_day(
        None,
        benchmark="CASH",
        signal_date=date(2025, 1, 2),
        trade_date=date(2025, 1, 3),
        execution_frequency="1min",
        dataset_identity_sha256="a" * 64,
        dataset_lineage_id="b" * 64,
        prior_nav=None,
    )

    assert result["status"] == "cash_baseline"
    assert result["benchmark_return"] == 0.0
    assert result["benchmark_wealth"] == 1.0


def test_pair_cash_baseline_preserves_the_prior_relative_wealth_chain() -> None:
    result = _benchmark_chain_day(
        {"this": "evidence is intentionally ignored for CASH"},
        benchmark="CASH",
        signal_date=date(2025, 1, 3),
        trade_date=date(2025, 1, 6),
        execution_frequency="1min",
        dataset_identity_sha256="a" * 64,
        dataset_lineage_id="b" * 64,
        prior_nav=SimpleNamespace(benchmark_wealth=1.0),
    )

    assert result["benchmark_return"] == 0.0
    assert result["benchmark_wealth"] == 1.0
