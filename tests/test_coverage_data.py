from datetime import date

import pytest

from quant_data.coverage_data import (
    COVERAGE_BUNDLES,
    DEFAULT_COVERAGE_BUNDLES,
    OPTIONAL_COVERAGE_BUNDLES,
    coverage_bundle_datasets,
    coverage_primary_key_candidates,
    coverage_secondary_specs,
)
from quant_data.supplemental_data import (
    bundle_datasets,
    next_pagination_specs,
    supplemental_specs,
)

pytestmark = pytest.mark.no_database


def test_coverage_inventory_matches_audited_default_and_optional_counts() -> None:
    default = set().union(
        *(coverage_bundle_datasets(bundle) for bundle in DEFAULT_COVERAGE_BUNDLES)
    )
    optional = set().union(
        *(coverage_bundle_datasets(bundle) for bundle in OPTIONAL_COVERAGE_BUNDLES)
    )
    assert len(default) == 59
    assert len(optional) == 26
    assert default.isdisjoint(optional)
    assert all(coverage_primary_key_candidates(dataset) for dataset in default | optional)


@pytest.mark.parametrize(
    ("dataset", "expected"),
    (
        ("broker_recommend", ("month", "broker", "ts_code")),
        ("ccass_hold_detail", ("ts_code", "trade_date", "col_participant_id")),
        ("daily_info", ("trade_date", "ts_code")),
        ("hm_detail", ("trade_date", "ts_code", "hm_name", "hm_orgs")),
        ("idx_anns", ("url",)),
        ("slb_len", ("trade_date",)),
        ("us_adjfactor", ("trade_date", "exchange", "ts_code")),
    ),
)
def test_coverage_primary_keys_match_provider_row_identity(
    dataset: str, expected: tuple[str, ...]
) -> None:
    assert coverage_primary_key_candidates(dataset)[0] == expected


@pytest.mark.parametrize("bundle", sorted(COVERAGE_BUNDLES))
def test_every_coverage_bundle_has_a_stable_task_inventory(bundle: str) -> None:
    assert bundle_datasets(bundle) == coverage_bundle_datasets(bundle)


@pytest.mark.parametrize("bundle", sorted(COVERAGE_BUNDLES - {"strategy_specialty_minutes"}))
def test_every_primary_coverage_request_is_resumable_and_truncation_safe(
    bundle: str,
) -> None:
    specs = supplemental_specs(
        bundle,
        start=date(2024, 1, 2),
        end=date(2024, 1, 2),
        trading_dates=["20240102"],
        max_attempts=5,
    )
    assert specs
    for spec in specs:
        assert spec.allow_empty is True
        assert spec.max_attempts == 5
        assert coverage_primary_key_candidates(spec.dataset)
        assert spec.scope["page_group"]
        assert int(spec.scope["page_size"]) > 0
        assert int(spec.scope["max_pages"]) > 1
        assert int(spec.scope["offset"]) == 0
        assert spec.params["limit"] == spec.scope["page_size"]
        assert spec.params["offset"] == 0


def test_default_rules_plan_full_market_cross_sections_without_stock_loops() -> None:
    for bundle in sorted(DEFAULT_COVERAGE_BUNDLES):
        specs = supplemental_specs(
            bundle,
            start=date(2024, 1, 2),
            end=date(2024, 1, 2),
            trading_dates=["20240102"],
            max_attempts=3,
        )
        datasets = {spec.dataset for spec in specs}
        expected = coverage_bundle_datasets(bundle)
        if bundle == "cn_governance_risk":
            expected = expected - {"stk_rewards"}
        assert datasets == expected
        assert all(
            "ts_code" not in spec.params for spec in specs if spec.dataset != "stock_company"
        )


def test_coverage_pagination_uses_rule_owned_page_ceiling() -> None:
    specs = supplemental_specs(
        "cn_capital_flow",
        start=date(2024, 1, 2),
        end=date(2024, 1, 2),
        trading_dates=["20240102"],
        max_attempts=3,
    )
    target = next(spec for spec in specs if spec.dataset == "moneyflow_dc")
    following = next_pagination_specs(
        specs,
        [
            {
                "unit_key": spec.unit_key,
                "row_count": spec.scope["page_size"] if spec == target else 0,
            }
            for spec in specs
        ],
    )
    page = next(spec for spec in following if spec.dataset == "moneyflow_dc")
    assert page.params["offset"] == 6_000
    assert page.scope["max_pages"] == 4


def test_capital_flow_uses_year_month_and_daily_grains_by_density() -> None:
    dates = ["20240102", "20241231", "20250102"]
    specs = supplemental_specs(
        "cn_capital_flow",
        start=date(2024, 1, 2),
        end=date(2025, 2, 2),
        trading_dates=dates,
        max_attempts=3,
    )

    assert len([spec for spec in specs if spec.dataset == "moneyflow_hsgt"]) == 2
    assert len([spec for spec in specs if spec.dataset == "moneyflow_mkt_dc"]) == 2
    for dataset in ("moneyflow_cnt_ths", "moneyflow_ind_ths", "moneyflow_ind_dc"):
        monthly = [spec for spec in specs if spec.dataset == dataset]
        assert len(monthly) == 14
        assert all(spec.scope["partition_axis"] == "date" for spec in monthly)
    for dataset in ("moneyflow_ths", "moneyflow_dc"):
        daily = [spec for spec in specs if spec.dataset == dataset]
        assert len(daily) == len(dates)
        assert all("trade_date" in spec.params for spec in daily)


def test_only_provider_mandated_symbol_paths_expand_by_symbol() -> None:
    governance = coverage_secondary_specs(
        "cn_governance_risk",
        {"stk_rewards": ["000001.SZ", "600000.SH"]},
        start=date(2024, 1, 1),
        end=date(2024, 1, 31),
        max_attempts=3,
    )
    assert len(governance) == 1
    assert governance[0].params["ts_code"] == "000001.SZ,600000.SH"
    assert governance[0].allow_empty is True
    assert governance[0].max_attempts == 3
    assert governance[0].scope["row_limit"] == 10_000

    minutes = coverage_secondary_specs(
        "strategy_specialty_minutes",
        {"sw_mins": ["801010.SI"], "hk_mins": ["00700.HK"]},
        start=date(2024, 1, 1),
        end=date(2024, 2, 2),
        max_attempts=3,
    )
    assert len(minutes) == 4
    assert {spec.params["freq"] for spec in minutes} == {"5min"}
    assert {spec.dataset for spec in minutes} == {"sw_mins", "hk_mins"}
    assert all(spec.allow_empty is True for spec in minutes)
    assert all(spec.max_attempts == 3 for spec in minutes)
    assert all(int(spec.scope["row_limit"]) > 0 for spec in minutes)
