from __future__ import annotations

import pytest

from quant_platform.worker import _paper_settlement_status

pytestmark = pytest.mark.no_database


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        (None, "awaiting_paper_account"),
        ({"status": "awaiting_simulation"}, "awaiting_paper_account"),
        (
            {"status": "active", "simulation_portfolio_id": None},
            "awaiting_paper_account",
        ),
        (
            {"status": "active", "simulation_portfolio_id": "paper-account-1"},
            "paper_validating",
        ),
    ],
)
def test_worker_reports_paper_stage_only_after_account_is_active(
    stage: dict | None,
    expected: str,
) -> None:
    assert _paper_settlement_status(stage) == expected
