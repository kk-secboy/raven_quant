from datetime import date

import pytest

from quant_data.catalog import ALL_DEFINITIONS
from quant_data.models import ProviderResult
from quant_data.provider import ProviderError
from quant_data.supplemental_data import (
    a_share_bulk_history_specs,
    bond_reference_specs,
    bundle_datasets,
    etf_constituent_history_specs,
    etf_constituent_overflow_repartition_specs,
    market_daily_specs,
    market_financial_specs,
    next_pagination_specs,
    require_pagination_terminated,
    share_float_overflow_repartition_specs,
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


def test_institutional_bundle_matches_its_declared_interface_contract() -> None:
    specs = supplemental_specs(
        "cn_institutional",
        start=date(2024, 1, 2),
        end=date(2024, 1, 2),
        trading_dates=["20240102"],
        max_attempts=3,
    )
    assert {spec.dataset for spec in specs} == {
        "report_rc",
        "etf_basic",
        "ci_daily",
        "shibor_quote",
        "major_news",
    }
    assert len([spec for spec in specs if spec.dataset == "major_news"]) == 9
    assert all("ts_code" not in spec.params for spec in specs)
    assert next(spec for spec in specs if spec.dataset == "report_rc").scope[
        "page_size"
    ] == 3_000
    assert next(spec for spec in specs if spec.dataset == "ci_daily").params[
        "trade_date"
    ] == "20240102"
    assert next(spec for spec in specs if spec.dataset == "major_news").scope[
        "page_size"
    ] == 400
    assert next(spec for spec in specs if spec.dataset == "major_news").fields == (
        "title",
        "content",
        "pub_time",
        "src",
    )
    assert bundle_datasets("cn_institutional") == {
        *{spec.dataset for spec in specs},
        "etf_sh_cons",
        "etf_sz_cons",
    }


def test_institutional_pages_advance_instead_of_accepting_provider_caps() -> None:
    specs = supplemental_specs(
        "cn_institutional",
        start=date(2024, 1, 2),
        end=date(2024, 1, 2),
        trading_dates=["20240102"],
        max_attempts=3,
    )
    target = next(spec for spec in specs if spec.dataset == "major_news")
    next_specs = next_pagination_specs(
        [target],
        [{"unit_key": target.unit_key, "row_count": 400}],
    )
    assert len(next_specs) == 1
    assert next_specs[0].params["offset"] == 400
    assert next_specs[0].fields == target.fields


def test_major_news_uses_monthly_source_windows_instead_of_daily_churn() -> None:
    specs = supplemental_specs(
        "cn_institutional",
        start=date(2024, 1, 15),
        end=date(2024, 3, 2),
        trading_dates=["20240115", "20240301"],
        max_attempts=3,
    )
    news = [spec for spec in specs if spec.dataset == "major_news"]

    assert len(news) == 27
    assert news[0].params["start_date"] == "2024-01-15 00:00:00"
    assert news[0].params["end_date"] == "2024-01-31 23:59:59"
    assert news[-1].params["start_date"] == "2024-03-01 00:00:00"
    assert news[-1].params["end_date"] == "2024-03-02 23:59:59"


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


def test_a_share_financial_history_includes_four_quarters_before_start() -> None:
    specs = a_share_bulk_history_specs(
        start=date(2024, 1, 1),
        end=date(2024, 1, 2),
        max_attempts=3,
    )
    periods = {
        spec.params["period"]
        for spec in specs
        if spec.dataset == "fina_indicator"
    }
    assert periods == {"20230331", "20230630", "20230930", "20231231"}


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
    basic_offsets = [spec.params["offset"] for spec in next_specs if spec.dataset == "opt_basic"]
    daily_offsets = [spec.params["offset"] for spec in next_specs if spec.dataset == "opt_daily"]
    assert basic_offsets == list(range(2_000, 18_000, 2_000))
    assert daily_offsets == [2_000]


def test_option_contract_master_can_continue_beyond_old_64_page_ceiling() -> None:
    specs = supplemental_specs(
        "cn_options_bonds",
        start=date(2024, 1, 2),
        end=date(2024, 1, 2),
        trading_dates=["20240102"],
        max_attempts=3,
    )
    specs = [spec for spec in specs if spec.dataset == "opt_basic"]
    while len(specs) < 65:
        rows = [
            {"unit_key": spec.unit_key, "row_count": int(spec.scope["page_size"])}
            for spec in specs
        ]
        specs.extend(next_pagination_specs(specs, rows))
    assert len(specs) == 65
    assert specs[-1].params["offset"] == 128_000


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
    share = next(spec for spec in specs if spec.dataset == "fund_share")
    assert share.params["start_date"] == "20240102"
    assert share.params["end_date"] == "20240102"
    assert share.scope["partition_axis"] == "date"
    masters = [spec for spec in specs if spec.dataset == "fund_basic"]
    assert {spec.params["market"] for spec in masters} == {"E", "O"}
    assert {spec.params["status"] for spec in masters} == {"L", "I", "D"}
    assert len(masters) == 6
    paged_master = next(
        spec
        for spec in masters
        if spec.params["market"] == "O" and spec.params["status"] == "L"
    )
    assert paged_master.params["limit"] == 5_000
    assert paged_master.params["offset"] == 0
    assert paged_master.scope["page_size"] == 5_000
    assert all(
        spec.scope["row_limit"] == 15_000
        for spec in masters
        if spec is not paged_master
    )
    etf_masters = [spec for spec in specs if spec.dataset == "etf_basic"]
    assert {spec.params["list_status"] for spec in etf_masters} == {"L", "D", "P"}
    assert next(spec for spec in specs if spec.dataset == "etf_index").scope["page_size"] == 5_000


def test_fund_share_uses_month_ranges_but_announcements_use_calendar_days() -> None:
    specs = supplemental_specs(
        "cn_funds",
        start=date(2024, 1, 5),
        end=date(2024, 2, 4),
        trading_dates=["20240105", "20240108", "20240202"],
        max_attempts=3,
    )

    shares = [spec for spec in specs if spec.dataset == "fund_share"]
    assert [(spec.params["start_date"], spec.params["end_date"]) for spec in shares] == [
        ("20240105", "20240131"),
        ("20240201", "20240204"),
    ]
    dividends = [spec for spec in specs if spec.dataset == "fund_div"]
    portfolios = [spec for spec in specs if spec.dataset == "fund_portfolio"]
    assert len(dividends) == 31
    assert len(portfolios) == 31
    assert {spec.params["ann_date"] for spec in dividends} >= {"20240106", "20240107"}


def test_dense_fund_portfolio_can_continue_beyond_the_old_256_page_ceiling() -> None:
    specs = supplemental_specs(
        "cn_funds",
        start=date(2024, 3, 29),
        end=date(2024, 3, 29),
        trading_dates=["20240329"],
        max_attempts=3,
    )
    specs = [spec for spec in specs if spec.dataset == "fund_portfolio"]
    rows = []
    for _ in range(256):
        current = specs[-1]
        rows.append(
            {"unit_key": current.unit_key, "row_count": int(current.scope["page_size"])}
        )
        specs.extend(next_pagination_specs(specs, rows))
    assert len(specs) == 257
    assert specs[-1].params["offset"] == 512_000


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


def test_us_financial_pagination_continues_beyond_old_64_page_ceiling() -> None:
    specs = market_financial_specs(
        "us",
        [],
        start=date(2024, 1, 1),
        end=date(2024, 3, 31),
        max_attempts=3,
    )
    pages = [spec for spec in specs if spec.dataset == "us_income"]
    rows: list[dict] = []
    for _ in range(9):
        current_window = sorted(pages, key=lambda item: int(item.scope["offset"]))[-8:]
        rows.extend(
            {"unit_key": page.unit_key, "row_count": int(page.scope["page_size"])}
            for page in current_window
        )
        pages.extend(next_pagination_specs(pages, rows))
    assert max(int(page.scope["offset"]) for page in pages) >= 71_000


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


def test_pagination_rejects_a_full_last_allowed_page(monkeypatch) -> None:
    from quant_data import supplemental_data

    specs = supplemental_specs(
        "cn_options_bonds",
        start=date(2024, 1, 2),
        end=date(2024, 1, 2),
        trading_dates=["20240102"],
        max_attempts=3,
    )
    specs = [spec for spec in specs if spec.dataset == "opt_basic"]
    monkeypatch.setitem(supplemental_data._PAGINATION_MAX_PAGES, "opt_basic", 2)
    rows = []
    for _ in range(2):
        current = specs[-1]
        rows.append({"unit_key": current.unit_key, "row_count": int(current.scope["page_size"])})
        specs.extend(next_pagination_specs(specs, rows))
    assert len(specs) == 2
    with pytest.raises(RuntimeError, match="pagination did not reach"):
        require_pagination_terminated(specs, rows)


def test_share_float_continues_after_the_old_64_page_ceiling() -> None:
    specs = [
        spec
        for spec in a_share_bulk_history_specs(
            start=date(2024, 1, 1),
            end=date(2024, 1, 31),
            max_attempts=3,
        )
        if spec.dataset == "share_float"
    ]
    rows: list[dict[str, object]] = []
    for _ in range(64):
        current = specs[-1]
        rows.append(
            {
                "unit_key": current.unit_key,
                "row_count": int(current.scope["page_size"]),
            }
        )
        specs.extend(next_pagination_specs(specs, rows))

    assert int(specs[-1].scope["offset"]) == 64_000


def test_share_float_offset_cap_repartitions_the_whole_month_by_day() -> None:
    parent = next(
        spec
        for spec in a_share_bulk_history_specs(
            start=date(2024, 1, 1),
            end=date(2024, 1, 31),
            max_attempts=3,
        )
        if spec.dataset == "share_float"
    )
    failed = parent
    for _ in range(101):
        failed = next_pagination_specs(
            [failed],
            [{"unit_key": failed.unit_key, "row_count": 1_000}],
        )[0]
    assert failed.params["offset"] == 101_000

    daily = share_float_overflow_repartition_specs(failed)
    assert len(daily) == 31
    assert daily[0].params == {
        "start_date": "20240101",
        "end_date": "20240101",
        "limit": 6_000,
        "offset": 0,
    }
    assert daily[-1].params["start_date"] == "20240131"
    assert all(
        item.scope["supersedes_page_group"] == parent.scope["page_group"]
        for item in daily
    )
    assert all(item.scope["expected_date"] == item.params["start_date"] for item in daily)

    following = next_pagination_specs(
        [parent, *daily],
        [
            {"unit_key": parent.unit_key, "row_count": 1_000},
            {"unit_key": daily[0].unit_key, "row_count": 6_000},
            *[
                {"unit_key": item.unit_key, "row_count": 0}
                for item in daily[1:]
            ],
        ],
    )
    assert len(following) == 1
    assert following[0].params["start_date"] == "20240101"
    assert following[0].params["offset"] == 6_000

    require_pagination_terminated(
        [parent, *daily],
        [
            {"unit_key": parent.unit_key, "row_count": 1_000},
            *[
                {"unit_key": item.unit_key, "row_count": 0}
                for item in daily
            ],
        ],
    )


def test_etf_constituent_history_starts_one_partition_per_active_symbol_range() -> None:
    specs = etf_constituent_history_specs(
        {
            "510050.SH": (date(2026, 1, 2), date(2026, 3, 16)),
            "159001.SZ": (date(2026, 2, 2), date(2026, 3, 16)),
            "512000.SH": (date(2026, 1, 2), date(2026, 3, 1)),
        },
        max_attempts=5,
    )

    assert len(specs) == 3
    assert {item.params["ts_code"] for item in specs} == {
        "510050.SH",
        "512000.SH",
        "159001.SZ",
    }
    shenzhen = next(item for item in specs if item.params["ts_code"] == "159001.SZ")
    assert shenzhen.params["start_date"] == "20260202"
    assert shenzhen.params["end_date"] == "20260316"
    assert all(item.scope["partition_axis"] == "date" for item in specs)


def test_market_daily_specs_use_only_calendar_open_dates() -> None:
    specs = market_daily_specs(
        "hk", ["20240102", "20240104"], max_attempts=3
    )

    assert {spec.dataset for spec in specs} == {"hk_daily", "hk_daily_adj"}
    assert {spec.params["trade_date"] for spec in specs} == {
        "20240102",
        "20240104",
    }


def test_hk_and_us_initial_market_plans_contain_only_master_and_calendar() -> None:
    hk = supplemental_specs(
        "hk_market",
        start=date(2024, 1, 1),
        end=date(2024, 1, 7),
        trading_dates=[],
        max_attempts=3,
    )
    us = supplemental_specs(
        "us_market",
        start=date(2024, 1, 1),
        end=date(2024, 1, 7),
        trading_dates=[],
        max_attempts=3,
    )

    assert {spec.dataset for spec in hk} == {"hk_basic", "hk_tradecal"}
    assert {
        spec.params["list_status"] for spec in hk if spec.dataset == "hk_basic"
    } == {"L", "D", "P"}
    assert {spec.dataset for spec in us} == {"us_basic", "us_tradecal"}


def test_etf_constituent_history_uses_symbol_date_windows() -> None:
    specs = etf_constituent_history_specs(
        {
            "510050.SH": (date(2026, 1, 2), date(2026, 7, 15)),
            "159001.SZ": (date(2026, 3, 2), date(2026, 7, 15)),
        },
        max_attempts=5,
    )

    assert len(specs) == 2
    shanghai = next(spec for spec in specs if spec.dataset == "etf_sh_cons")
    assert shanghai.params == {
        "ts_code": "510050.SH",
        "start_date": "20260102",
        "end_date": "20260715",
        "limit": 3_000,
        "offset": 0,
    }
    assert shanghai.scope["expected_date_start"] == "20260102"
    assert shanghai.scope["expected_date_end"] == "20260715"


def test_etf_symbol_window_offset_cap_bisects_the_date_range() -> None:
    failed = etf_constituent_history_specs(
        {"510050.SH": (date(2026, 1, 2), date(2026, 7, 15))},
        max_attempts=5,
    )[0]
    for _ in range(34):
        failed = next_pagination_specs(
            [failed],
            [{"unit_key": failed.unit_key, "row_count": 3_000}],
        )[0]
    assert failed.params["offset"] == 102_000

    children = etf_constituent_overflow_repartition_specs(failed)
    assert len(children) == 2
    assert all(child.params["ts_code"] == "510050.SH" for child in children)
    assert children[0].params["start_date"] == "20260102"
    assert children[-1].params["end_date"] == "20260715"
    assert children[0].params["end_date"] < children[1].params["start_date"]
    assert all(
        child.scope["supersedes_page_group"] == failed.scope["page_group"]
        for child in children
    )


def test_share_float_has_a_natural_date_and_duplicate_gate() -> None:
    definition = ALL_DEFINITIONS["share_float"]
    assert definition.date_field == "float_date"
    assert definition.primary_key == (
        "ts_code",
        "ann_date",
        "float_date",
        "holder_name",
        "share_type",
    )
