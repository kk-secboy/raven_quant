from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.no_database

ROOT = Path(__file__).resolve().parents[1]
PLATFORM_ROOT = ROOT / "src" / "quant_platform"


def test_retired_execution_modules_are_deleted() -> None:
    retired_modules = (
        "broker_gateway.py",
        "pair_portfolio_store.py",
        "paper_trading.py",
        "portfolio_store.py",
    )

    for module_name in retired_modules:
        assert not (PLATFORM_ROOT / module_name).exists()


def test_active_platform_does_not_restore_legacy_execution_paths() -> None:
    # retention.py may read the frozen historical tables so their source data is
    # not deleted. It is intentionally excluded because it cannot execute or
    # write any legacy order, fill, broker, or rebalance flow.
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in PLATFORM_ROOT.rglob("*.py")
        if path.name != "retention.py"
    )
    forbidden = (
        "paper_rebalance",
        "pair_paper_rebalance",
        "run_pair_paper_step",
        "paper_orders",
        "paper_fills",
        "pair_paper_orders",
        "pair_paper_fills",
        "quant_platform.broker_gateway",
        '"/api/broker',
        '"/api/pair-portfolios',
    )

    for marker in forbidden:
        assert marker not in sources
