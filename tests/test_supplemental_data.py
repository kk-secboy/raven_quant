from datetime import date

import pytest

from quant_data.models import ProviderResult
from quant_data.provider import ProviderError
from quant_data.supplemental_data import (
    a_share_bulk_history_specs,
    bond_reference_specs,
    bundle_datasets,
    market_financial_specs,
    next_pagination_specs,
    require_pagination_terminated,
    supplemental_specs,
    validate_supplemental,
)

pytestmark = pytest.mark.no_database


def test_macro_bundle_is_small_and_contains_verified_interfaces() -> None:
    specs = supplemental_specs(
        "cn_macro",
        start=date(2024, 1, 1),
        end=date(2026, 7, 13),
        trading_dates=[],
        max_attempts=3,
    )
    assert {spec.api_name for spec in specs} == {
        "cn_gdp",
        "cn_cpi",
        "cn_ppi",
        "cn_pmi",
        "cn_m",
        "sf_month",
        "shibor",
        "shibor_lpr",
        "cn_schedule",
    }
    schedules = [spec for spec in specs if spec.dataset == "cn_schedule"]
    assert schedules[0].params == {"m": "202401"}
    assert schedules[-1].params == {"m": "202607"}
    assert bundle_datasets("cn_macro") == {spec.dataset for spec in specs}


def test_extended_bundle_adds_point_in_time_st_and_sw_industry_bars() -> None:
    specs = supplemental_specs(
        "cn_extended_daily",
        start=date(2024, 1, 2),
        end=date(2024, 1, 2),
        trading_dates=["20240102"],
        max_attempts=3,
    )
    institutional = [spec for spec in specs if spec.dataset in {"stock_st", "sw_daily"}]
    assert {spec.api_name for spec in institutional} == {"stock_st", "sw_daily"}
    assert {spec.params["trade_date"] for spec in institutional} == {"20240102"}
    assert {spec.scope["page_size"] for spec in institutional} == {1_000, 4_000}


def test_a_share_financial_specs_use_cross_sectional_vip_batches() -> None:
    specs = supplemental_specs(
        "cn_extended_daily",
        start=date(2024, 1, 1),
        end=date(2026, 7, 13),
        trading_dates=[],
        max_attempts=3,
    )
    audits = [spec for spec in specs if spec.dataset == "fina_audit"]
    main_business = [spec for spec in specs if spec.dataset == "fina_mainbz"]
    assert audits
    assert main_business
    assert {spec.api_name for spec in audits} == {"fina_audit_vip"}
    assert {spec.api_name for spec in main_business} == {"fina_mainbz_vip"}
    assert all("ts_code" not in spec.params for spec in [*audits, *main_business])
    assert {spec.params["type"] for spec in main_business} == {"P", "D", "I"}
    assert all(spec.scope["page_size"] == 1_000 for spec in [*audits, *main_business])
    assert {"fina_audit", "fina_mainbz"} <= bundle_datasets("cn_extended_daily")


def test_full_a_share_history_uses_market_cross_sections_instead_of_symbols() -> None:
    specs = a_share_bulk_history_specs(
        start=date(2024, 1, 1),
        end=date(2024, 1, 2),
        max_attempts=3,
    )
    assert len({spec.unit_key for spec in specs}) == len(specs)
    assert all("ts_code" not in spec.params for spec in specs)
    financial_apis = {
        spec.api_name
        for spec in specs
        if spec.dataset
        in {"income", "balancesheet", "cashflow", "fina_indicator", "forecast", "express"}
    }
    assert financial_apis == {
        "income_vip",
        "balancesheet_vip",
        "cashflow_vip",
        "fina_indicator_vip",
        "forecast_vip",
        "express_vip",
    }
    event_datasets = {
        "namechange",
        "dividend",
        "repurchase",
        "share_float",
        "pledge_stat",
        "pledge_detail",
        "stk_holdertrade",
        "anns_d",
    }
    assert event_datasets <= {spec.dataset for spec in specs}
    assert all(spec.scope["page_size"] == 1_000 for spec in specs)


def test_options_bundle_starts_one_page_per_partition_and_advances_full_pages() -> None:
    specs = supplemental_specs(
        "cn_options_bonds",
        start=date(2024, 1, 2),
        end=date(2024, 1, 2),
        trading_dates=["20240102"],
        max_attempts=3,
    )
    assert len({spec.unit_key for spec in specs}) == len(specs)
    assert all(spec.params["offset"] == 0 for spec in specs)
    assert all(spec.allow_empty for spec in specs)
    rows = [
        {
            "unit_key": spec.unit_key,
            "row_count": (
                int(spec.scope["page_size"]) if spec.dataset in {"opt_basic", "opt_daily"} else 0
            ),
        }
        for spec in specs
    ]
    next_specs = next_pagination_specs(specs, rows)
    assert {spec.dataset for spec in next_specs} == {"opt_basic", "opt_daily"}
    assert all(spec.params["offset"] == spec.scope["page_size"] for spec in next_specs)


def test_fund_bundle_covers_research_master_and_time_series() -> None:
    specs = supplemental_specs(
        "cn_funds",
        start=date(2024, 1, 2),
        end=date(2024, 1, 2),
        trading_dates=["20240102"],
        max_attempts=3,
    )
    assert {spec.dataset for spec in specs} == {
        "fund_basic",
        "fund_company",
        "fund_manager",
        "fund_nav",
        "fund_share",
        "fund_div",
        "fund_portfolio",
        "etf_basic",
        "etf_index",
    }
    nav = next(spec for spec in specs if spec.dataset == "fund_nav")
    assert nav.params["nav_date"] == "20240102"
    masters = [spec for spec in specs if spec.dataset == "fund_basic"]
    assert {spec.params["market"] for spec in masters} == {"E", "O"}
    assert {spec.params["status"] for spec in masters} == {"L", "I", "D"}
    assert len(masters) == 6
    assert all(spec.scope["row_limit"] == 15_000 for spec in masters)
    etf_masters = [spec for spec in specs if spec.dataset == "etf_basic"]
    assert {spec.params["list_status"] for spec in etf_masters} == {"L", "D", "P"}
    assert next(spec for spec in specs if spec.dataset == "etf_index").scope["page_size"] == 5_000


def test_futures_bundle_includes_calendar_and_continuous_mapping() -> None:
    specs = supplemental_specs(
        "cn_futures",
        start=date(2024, 1, 2),
        end=date(2024, 1, 2),
        trading_dates=["20240102"],
        max_attempts=3,
    )
    assert {"fut_trade_cal", "fut_mapping", "ft_limit"} <= {spec.dataset for spec in specs}
    mapping = next(spec for spec in specs if spec.dataset == "fut_mapping")
    assert mapping.params["trade_date"] == "20240102"
    limit = next(spec for spec in specs if spec.dataset == "ft_limit")
    assert limit.params["trade_date"] == "20240102"
    assert limit.scope["page_size"] == 4_000


def test_options_bonds_bundle_covers_convertible_bond_lifecycle_and_curve() -> None:
    specs = supplemental_specs(
        "cn_options_bonds",
        start=date(2024, 1, 2),
        end=date(2024, 1, 2),
        trading_dates=["20240102"],
        max_attempts=3,
    )
    references = bond_reference_specs(
        ["110001.SH", "123001.SZ"],
        start=date(2024, 1, 2),
        end=date(2024, 1, 2),
        max_attempts=3,
    )
    assert {
        "cb_issue",
        "cb_redeem",
        "cb_rate",
        "cb_price_chg",
        "cb_share",
        "cb_rating",
        "top10_cb_holders",
        "yc_cb",
    } <= {spec.dataset for spec in [*specs, *references]}
    redeem = next(spec for spec in specs if spec.dataset == "cb_redeem")
    assert redeem.api_name == "cb_call"
    assert all(spec.params.get("ts_code") for spec in references)


def test_bond_reference_batches_keep_conversion_results_below_provider_limit() -> None:
    symbols = [f"{index:06d}.SH" for index in range(1, 102)]
    specs = bond_reference_specs(
        symbols,
        start=date(2024, 1, 1),
        end=date(2026, 7, 13),
        max_attempts=3,
    )
    shares = [spec for spec in specs if spec.dataset == "cb_share"]
    assert len(shares) == 11
    assert all(len(spec.params["ts_code"].split(",")) <= 10 for spec in shares)
    assert all(spec.params["start_date"] == "20240101" for spec in shares)


def test_global_bundle_covers_index_and_fx() -> None:
    specs = supplemental_specs(
        "global_markets",
        start=date(2024, 1, 2),
        end=date(2024, 1, 2),
        trading_dates=[],
        max_attempts=3,
    )
    assert {spec.dataset for spec in specs} == {
        "fx_obasic",
        "fx_daily",
        "index_global",
        "us_tycr",
    }
    fx = next(spec for spec in specs if spec.dataset == "fx_daily")
    assert fx.scope["page_size"] == 1_000
    treasury = next(spec for spec in specs if spec.dataset == "us_tycr")
    assert treasury.params == {"start_date": "20240102", "end_date": "20240102"}


def test_us_treasury_curve_is_partitioned_by_year() -> None:
    specs = supplemental_specs(
        "global_markets",
        start=date(2024, 7, 1),
        end=date(2026, 7, 13),
        trading_dates=[],
        max_attempts=3,
    )
    curves = [spec for spec in specs if spec.dataset == "us_tycr"]
    assert [spec.params for spec in curves] == [
        {"start_date": "20240701", "end_date": "20241231"},
        {"start_date": "20250101", "end_date": "20251231"},
        {"start_date": "20260101", "end_date": "20260713"},
    ]
    assert all(spec.scope["row_limit"] == 2_000 for spec in curves)


def test_dense_production_partitions_can_paginate_past_old_ceiling() -> None:
    specs = supplemental_specs(
        "cn_futures",
        start=date(2024, 1, 2),
        end=date(2024, 1, 2),
        trading_dates=["20240102"],
        max_attempts=3,
    )
    pages = [spec for spec in specs if spec.dataset == "fut_holding"]
    rows: list[dict] = []
    for _ in range(9):
        current = pages[-1]
        rows.append({"unit_key": current.unit_key, "row_count": int(current.scope["page_size"])})
        pages.extend(next_pagination_specs(pages, rows))
    assert len(pages) == 10
    assert pages[-1].scope["offset"] == 9_000


def test_market_financial_specs_are_symbol_scoped() -> None:
    specs = market_financial_specs(
        "hk",
        ["00700.HK", "00941.HK", "00700.HK"],
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
        max_attempts=3,
    )
    assert len(specs) == 8
    assert {spec.dataset for spec in specs} == {
        "hk_income",
        "hk_balancesheet",
        "hk_cashflow",
        "hk_fina_indicator",
    }
    assert {spec.params["ts_code"] for spec in specs} == {"00700.HK", "00941.HK"}


def test_us_financial_specs_use_period_cross_sections() -> None:
    specs = market_financial_specs(
        "us",
        ["AAPL", "MSFT"],
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
        max_attempts=3,
    )
    statements = [spec for spec in specs if spec.dataset != "us_fina_indicator"]
    indicators = [spec for spec in specs if spec.dataset == "us_fina_indicator"]
    assert all("ts_code" not in spec.params for spec in statements)
    assert {spec.api_name for spec in statements} == {
        "us_income_vip",
        "us_balancesheet_vip",
        "us_cashflow_vip",
    }
    assert all(spec.scope["page_size"] == 1_000 for spec in statements)
    assert {spec.params["ts_code"] for spec in indicators} == {"AAPL", "MSFT"}
    assert all(spec.scope["row_limit"] == 200 for spec in indicators)


def test_market_financial_row_limit_is_enforced_by_shared_validator() -> None:
    from quant_data.execution_data import validate_and_normalize

    spec = next(
        spec
        for spec in market_financial_specs(
            "us",
            ["AAPL"],
            start=date(2024, 1, 1),
            end=date(2024, 12, 31),
            max_attempts=3,
        )
        if spec.dataset == "us_fina_indicator"
    )
    rows = [{"ts_code": "AAPL"}] * 200
    with pytest.raises(ProviderError, match="may be truncated"):
        validate_and_normalize(
            spec,
            ProviderResult(spec.api_name, ["ts_code"], rows, b"{}"),
        )


def test_supplemental_validation_rejects_an_ignored_page_size() -> None:
    spec = supplemental_specs(
        "cn_options_bonds",
        start=date(2024, 1, 2),
        end=date(2024, 1, 2),
        trading_dates=["20240102"],
        max_attempts=3,
    )[0]
    rows = [{"ts_code": str(index)} for index in range(int(spec.scope["page_size"]) + 1)]
    with pytest.raises(ProviderError, match="ignored the requested page size"):
        validate_supplemental(spec, ProviderResult(spec.api_name, ["ts_code"], rows, b"{}"))


def test_pagination_requires_a_short_terminal_page() -> None:
    specs = supplemental_specs(
        "cn_options_bonds",
        start=date(2024, 1, 2),
        end=date(2024, 1, 2),
        trading_dates=["20240102"],
        max_attempts=3,
    )
    rows = [
        {"unit_key": spec.unit_key, "row_count": spec.scope.get("page_size", 1)} for spec in specs
    ]
    with pytest.raises(RuntimeError, match="pagination did not reach"):
        require_pagination_terminated(specs, rows)


def test_pagination_accepts_a_short_nonempty_terminal_page() -> None:
    specs = supplemental_specs(
        "cn_options_bonds",
        start=date(2024, 1, 2),
        end=date(2024, 1, 2),
        trading_dates=["20240102"],
        max_attempts=3,
    )
    rows = [
        {
            "unit_key": spec.unit_key,
            "row_count": min(400, int(spec.scope["page_size"]) - 1),
        }
        for spec in specs
    ]
    require_pagination_terminated(specs, rows)
    assert next_pagination_specs(specs, rows) == []


def test_pagination_rejects_a_full_last_allowed_page() -> None:
    specs = supplemental_specs(
        "cn_options_bonds",
        start=date(2024, 1, 2),
        end=date(2024, 1, 2),
        trading_dates=["20240102"],
        max_attempts=3,
    )
    specs = [spec for spec in specs if spec.dataset == "opt_basic"]
    rows = []
    for _ in range(64):
        current = specs[-1]
        rows.append({"unit_key": current.unit_key, "row_count": int(current.scope["page_size"])})
        specs.extend(next_pagination_specs(specs, rows))
    assert len(specs) == 64
    with pytest.raises(RuntimeError, match="pagination did not reach"):
        require_pagination_terminated(specs, rows)
