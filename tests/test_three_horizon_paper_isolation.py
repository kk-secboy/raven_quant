import pytest

from quant_platform.promotion import _autopilot_paper_accounts_compete


@pytest.mark.no_database
@pytest.mark.parametrize(
    ("current", "prior", "expected"),
    [
        ("short_1_5d", "short_1_5d", True),
        ("swing_1_6m", "swing_1_6m", True),
        ("long_1_3y", "long_1_3y", True),
        ("short_1_5d", "swing_1_6m", False),
        ("short_1_5d", "long_1_3y", False),
        ("swing_1_6m", "long_1_3y", False),
        ("legacy_ambiguous", "short_1_5d", True),
        ("long_1_3y", "legacy_ambiguous", True),
        (None, None, True),
    ],
)
def test_autopilot_paper_accounts_only_compete_inside_one_explicit_horizon(
    current: str | None,
    prior: str | None,
    expected: bool,
) -> None:
    assert _autopilot_paper_accounts_compete(current, prior) is expected
