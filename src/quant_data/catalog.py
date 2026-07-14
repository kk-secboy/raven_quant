from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DatasetDefinition:
    name: str
    api_name: str
    fields: tuple[str, ...] = ()
    allow_empty: bool = False
    date_field: str | None = "trade_date"
    primary_key: tuple[str, ...] = ()


REFERENCE_FIELDS = {
    "stock_basic": (
        "ts_code",
        "symbol",
        "name",
        "area",
        "industry",
        "market",
        "exchange",
        "list_status",
        "list_date",
        "delist_date",
        "is_hs",
    ),
    "trade_cal": ("exchange", "cal_date", "is_open", "pretrade_date"),
}

CORE_DAILY: tuple[DatasetDefinition, ...] = (
    DatasetDefinition(
        "daily",
        "daily",
        (
            "ts_code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "change",
            "pct_chg",
            "vol",
            "amount",
        ),
        primary_key=("ts_code", "trade_date"),
    ),
    DatasetDefinition(
        "adj_factor",
        "adj_factor",
        ("ts_code", "trade_date", "adj_factor"),
        primary_key=("ts_code", "trade_date"),
    ),
    DatasetDefinition(
        "daily_basic",
        "daily_basic",
        (
            "ts_code",
            "trade_date",
            "close",
            "turnover_rate",
            "turnover_rate_f",
            "volume_ratio",
            "pe",
            "pe_ttm",
            "pb",
            "ps",
            "ps_ttm",
            "dv_ratio",
            "dv_ttm",
            "total_share",
            "float_share",
            "free_share",
            "total_mv",
            "circ_mv",
        ),
        primary_key=("ts_code", "trade_date"),
    ),
    DatasetDefinition("suspend_d", "suspend_d", allow_empty=True),
    DatasetDefinition("stk_limit", "stk_limit", allow_empty=False),
    DatasetDefinition("limit_list_d", "limit_list_d", allow_empty=True),
)

RESEARCH_DAILY: tuple[DatasetDefinition, ...] = (
    DatasetDefinition("moneyflow", "moneyflow", allow_empty=True),
    DatasetDefinition("margin_detail", "margin_detail", allow_empty=True),
    DatasetDefinition("hsgt_top10", "hsgt_top10", allow_empty=True),
    DatasetDefinition("top_list", "top_list", allow_empty=True),
    DatasetDefinition("top_inst", "top_inst", allow_empty=True),
)

ETF_DAILY: tuple[DatasetDefinition, ...] = (
    DatasetDefinition(
        "fund_daily",
        "fund_daily",
        primary_key=("ts_code", "trade_date"),
    ),
    DatasetDefinition(
        "fund_adj",
        "fund_adj",
        primary_key=("ts_code", "trade_date"),
    ),
)

FUNDAMENTALS: tuple[DatasetDefinition, ...] = (
    DatasetDefinition("income", "income", allow_empty=True, date_field="ann_date"),
    DatasetDefinition("balancesheet", "balancesheet", allow_empty=True, date_field="ann_date"),
    DatasetDefinition("cashflow", "cashflow", allow_empty=True, date_field="ann_date"),
    DatasetDefinition("fina_indicator", "fina_indicator", allow_empty=True, date_field="ann_date"),
    DatasetDefinition("forecast", "forecast", allow_empty=True, date_field="ann_date"),
    DatasetDefinition("express", "express", allow_empty=True, date_field="ann_date"),
)

CORPORATE_EVENTS: tuple[DatasetDefinition, ...] = (
    DatasetDefinition("namechange", "namechange", allow_empty=True, date_field="ann_date"),
    DatasetDefinition("dividend", "dividend", allow_empty=True, date_field="ann_date"),
    DatasetDefinition("repurchase", "repurchase", allow_empty=True, date_field="ann_date"),
    DatasetDefinition("share_float", "share_float", allow_empty=True, date_field="ann_date"),
    DatasetDefinition("pledge_stat", "pledge_stat", allow_empty=True, date_field="end_date"),
    DatasetDefinition("pledge_detail", "pledge_detail", allow_empty=True, date_field="ann_date"),
    DatasetDefinition(
        "stk_holdertrade",
        "stk_holdertrade",
        allow_empty=True,
        date_field="ann_date",
    ),
    DatasetDefinition("anns_d", "anns_d", allow_empty=True, date_field="ann_date"),
)

INDUSTRY_CATALOG: tuple[DatasetDefinition, ...] = (
    DatasetDefinition("index_classify", "index_classify", date_field=None),
)

DISCLOSURE_FIELDS = (
    "ts_code",
    "ann_date",
    "end_date",
    "pre_date",
    "actual_date",
    "modify_date",
)

INDEX_CODES = ("000016.SH", "000300.SH", "000905.SH", "000852.SH")

ALL_DEFINITIONS = {
    definition.name: definition
    for definition in (
        *CORE_DAILY,
        *RESEARCH_DAILY,
        *ETF_DAILY,
        *FUNDAMENTALS,
        *CORPORATE_EVENTS,
        *INDUSTRY_CATALOG,
    )
}
