from __future__ import annotations

from datetime import date

import pytest

from quant_data.history_bounds import clip_history_range, history_start_date

pytestmark = pytest.mark.no_database


def test_report_rc_uses_documented_2010_history_boundary() -> None:
    assert history_start_date("report_rc") == date(2010, 1, 1)
    assert clip_history_range(
        "report_rc", date(2008, 1, 1), date(2026, 8, 3)
    ) == (date(2010, 1, 1), date(2026, 8, 3))
