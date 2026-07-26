from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from .models import FetchSpec
from .partitioning import partition_metadata
from .planner import compact_date
from .reference_data import apply_reference_refresh

DEFAULT_COVERAGE_BUNDLES = {
    "cn_governance_risk",
    "cn_capital_flow",
    "cn_fund_index_enhanced",
    "cn_derivatives_enhanced",
    "global_rates_enhanced",
    "research_corpus",
}
OPTIONAL_COVERAGE_BUNDLES = {"strategy_specialty", "strategy_specialty_minutes"}
COVERAGE_BUNDLES = DEFAULT_COVERAGE_BUNDLES | OPTIONAL_COVERAGE_BUNDLES
_ADAPTIVE_CAPITAL_FLOW_DATASETS = {
    "moneyflow_hsgt",
    "moneyflow_cnt_ths",
    "moneyflow_ind_ths",
    "moneyflow_ind_dc",
    "moneyflow_mkt_dc",
}


@dataclass(frozen=True, slots=True)
class CoverageRule:
    dataset: str
    mode: str
    page_size: int
    max_pages: int
    date_param: str = "trade_date"
    date_field: str = "trade_date"
    variants: tuple[tuple[tuple[str, Any], ...], ...] = ((),)


def _variants(**name_values: tuple[Any, ...]) -> tuple[tuple[tuple[str, Any], ...], ...]:
    result: list[tuple[tuple[str, Any], ...]] = [()]
    for name, values in name_values.items():
        result = [(*current, (name, value)) for current in result for value in values]
    return tuple(result)


_RULES: dict[str, tuple[CoverageRule, ...]] = {
    "cn_governance_risk": (
        CoverageRule("st", "calendar_daily", 1_000, 4, "pub_date", "pub_date"),
        CoverageRule(
            "stock_hsgt",
            "daily",
            2_000,
            4,
            variants=_variants(type=("HK_SZ", "SZ_HK", "HK_SH", "SH_HK")),
        ),
        CoverageRule(
            "stock_company",
            "once",
            4_000,
            4,
            variants=_variants(exchange=("SSE", "SZSE", "BSE")),
        ),
        CoverageRule("stk_managers", "calendar_daily", 2_000, 8, "ann_date", "ann_date"),
        CoverageRule("bse_mapping", "once", 1_000, 2),
        CoverageRule("ggt_top10", "daily", 2_000, 8),
        CoverageRule("ggt_daily", "daily", 1_000, 2),
        CoverageRule("stk_shock", "daily", 1_000, 8),
        CoverageRule("stk_high_shock", "daily", 1_000, 8),
        # stk_alert rows carry start_date/end_date (alert effective period), not
        # trade_date; validate against start_date or every row fails the date check.
        CoverageRule("stk_alert", "daily", 1_000, 8, "trade_date", "start_date"),
        CoverageRule("cyq_perf", "daily", 6_000, 4),
        CoverageRule("cyq_chips", "daily", 6_000, 4),
        CoverageRule("ccass_hold", "daily", 5_000, 8),
        CoverageRule("ccass_hold_detail", "daily", 6_000, 8),
        CoverageRule("broker_recommend", "month", 1_000, 8, "month", "ann_date"),
        CoverageRule("margin", "daily", 4_000, 2),
        CoverageRule("slb_len", "daily", 5_000, 4),
    ),
    "cn_capital_flow": (
        CoverageRule("moneyflow_hsgt", "year", 300, 4),
        CoverageRule("moneyflow_ths", "daily", 5_000, 4),
        CoverageRule("moneyflow_dc", "daily", 6_000, 4),
        CoverageRule("moneyflow_cnt_ths", "month_range", 5_000, 4),
        CoverageRule("moneyflow_ind_ths", "month_range", 5_000, 4),
        CoverageRule("moneyflow_ind_dc", "month_range", 5_000, 4),
        CoverageRule("moneyflow_mkt_dc", "year", 3_000, 4),
    ),
    "cn_fund_index_enhanced": (
        CoverageRule("etf_share_size", "daily", 5_000, 4),
        CoverageRule("idx_anns", "calendar_daily", 1_000, 8, "ann_date", "ann_date"),
        CoverageRule("ci_index_member", "once", 5_000, 16),
        CoverageRule("idx_factor_pro", "daily", 8_000, 4),
        CoverageRule("daily_info", "daily", 4_000, 4),
        CoverageRule("sz_daily_info", "daily", 4_000, 4),
        CoverageRule("mkt_idx_bmk", "once", 500, 4),
        CoverageRule("fund_factor_pro", "daily", 8_000, 4),
    ),
    "cn_derivatives_enhanced": (
        CoverageRule("fut_index_daily", "daily", 8_000, 4),
        CoverageRule("fut_weekly_detail", "year_week", 4_000, 8, "week", "week"),
        CoverageRule("sge_basic", "once", 100, 2),
        CoverageRule("sge_daily", "daily", 4_000, 4),
        CoverageRule("cb_factor_pro", "daily", 10_000, 4),
        CoverageRule("bc_otcqt", "daily", 2_000, 8),
        CoverageRule("bc_bestotcqt", "daily", 2_000, 8),
        CoverageRule("bond_blk", "daily", 1_000, 8),
        CoverageRule("bond_blk_detail", "daily", 1_000, 8),
        CoverageRule("eco_cal", "calendar_daily", 100, 8, "date", "date"),
    ),
    "global_rates_enhanced": (
        CoverageRule("hk_adjfactor", "daily", 6_000, 4),
        CoverageRule("us_adjfactor", "daily", 15_000, 4),
        CoverageRule("libor", "year", 4_000, 8),
        CoverageRule("hibor", "year", 4_000, 8),
        CoverageRule("us_trycr", "year", 2_000, 8),
        CoverageRule("us_tbr", "year", 2_000, 8),
        CoverageRule("us_tltr", "year", 2_000, 8),
        CoverageRule("us_trltr", "year", 2_000, 8),
    ),
    "research_corpus": (
        CoverageRule("npr", "calendar_daily_range", 500, 8, date_field="pub_time"),
        CoverageRule("research_report", "calendar_daily", 1_000, 8),
        CoverageRule("monetary_policy", "once", 1_000, 2, date_field="pub_date"),
        CoverageRule("cctv_news", "calendar_daily", 1_000, 2, "date", "date"),
        CoverageRule("irm_qa_sh", "daily", 3_000, 8),
        CoverageRule("irm_qa_sz", "daily", 3_000, 8),
        CoverageRule("wc_list", "once", 3_000, 2, date_field="pub_time"),
        CoverageRule("wc_cnt", "month_range", 3_000, 16, date_field="publish_time"),
    ),
    "strategy_specialty": (
        CoverageRule(
            "stk_nineturn", "daily", 10_000, 4, variants=_variants(freq=("D",))
        ),
        CoverageRule("stk_ah_comparison", "daily", 1_000, 4),
        CoverageRule("limit_list_ths", "daily", 4_000, 4),
        CoverageRule("limit_step", "daily", 2_000, 8),
        CoverageRule("limit_cpt_list", "daily", 2_000, 8),
        CoverageRule("ths_index", "once", 5_000, 4),
        CoverageRule("ths_daily", "daily", 3_000, 8),
        CoverageRule("ths_member", "once", 5_000, 16),
        CoverageRule("dc_index", "daily", 5_000, 8),
        CoverageRule("dc_member", "daily", 5_000, 8),
        CoverageRule("dc_daily", "daily", 2_000, 8),
        CoverageRule("hm_list", "once", 1_000, 2),
        CoverageRule("hm_detail", "daily", 2_000, 8),
        CoverageRule("ths_hot", "daily", 2_000, 8),
        CoverageRule("dc_hot", "daily", 2_000, 8),
        CoverageRule("tdx_index", "daily", 1_000, 8),
        CoverageRule("tdx_member", "daily", 3_000, 8),
        CoverageRule("tdx_daily", "daily", 3_000, 8),
        CoverageRule("kpl_list", "daily", 8_000, 4),
        CoverageRule("kpl_concept_cons", "daily", 3_000, 8),
        CoverageRule("dc_concept", "daily", 5_000, 4),
        CoverageRule("dc_concept_cons", "daily", 5_000, 8),
        CoverageRule("wz_index", "calendar_daily", 1_000, 2, "date", "date"),
        CoverageRule("gz_index", "calendar_daily", 1_000, 2, "date", "date"),
    ),
    "strategy_specialty_minutes": (),
}


SECONDARY_DATASETS = {
    "cn_governance_risk": {"stk_rewards"},
    "strategy_specialty_minutes": {"sw_mins", "hk_mins"},
}
COVERAGE_DATASETS = frozenset(
    dataset
    for bundle in COVERAGE_BUNDLES
    for dataset in (
        {rule.dataset for rule in _RULES[bundle]}
        | SECONDARY_DATASETS.get(bundle, set())
    )
)

_PRIMARY_KEY_OVERRIDES: dict[str, tuple[tuple[str, ...], ...]] = {
    "st": (("ts_code", "pub_date", "imp_date", "st_type"),),
    "stock_hsgt": (("ts_code", "trade_date", "type"),),
    "stock_company": (("ts_code",),),
    "stk_managers": (("ts_code", "ann_date", "name", "title"),),
    "stk_rewards": (("ts_code", "ann_date", "name", "title"),),
    "bse_mapping": (("o_code", "n_code"),),
    "ggt_top10": (("ts_code", "trade_date", "market_type"),),
    "ggt_daily": (("trade_date",),),
    "cyq_chips": (("ts_code", "trade_date", "price"),),
    "ccass_hold_detail": (
        ("ts_code", "trade_date", "broker_id"),
        ("hk_code", "trade_date", "broker_id"),
    ),
    "broker_recommend": (("month", "ts_code"), ("ann_date", "ts_code")),
    "margin": (("trade_date", "exchange_id"),),
    "moneyflow_hsgt": (("trade_date",),),
    "moneyflow_mkt_dc": (("trade_date",),),
    "idx_anns": (("ann_date", "ts_code", "title"), ("ann_date", "title")),
    "ci_index_member": (("l3_code", "ts_code", "in_date"), ("l3_code", "ts_code")),
    "daily_info": (("trade_date", "exchange"),),
    "sz_daily_info": (("trade_date", "ts_code"), ("trade_date", "name")),
    "mkt_idx_bmk": (("ts_code", "bmk_type", "bmk_level"),),
    "fut_weekly_detail": (("week", "exchange", "prd"),),
    "sge_basic": (("ts_code",),),
    "eco_cal": (("date", "currency", "event"), ("date", "country", "event")),
    "libor": (("date", "curr_type"),),
    "hibor": (("date",),),
    "us_trycr": (("date",),),
    "us_tbr": (("date",),),
    "us_tltr": (("date",),),
    "us_trltr": (("date",),),
    "npr": (("pub_time", "title"), ("datetime", "title")),
    "research_report": (("trade_date", "ts_code", "title"),),
    "monetary_policy": (("pub_date", "title"),),
    "cctv_news": (("date", "title"),),
    "irm_qa_sh": (("trade_date", "ts_code", "q"), ("trade_date", "ts_code", "question")),
    "irm_qa_sz": (("trade_date", "ts_code", "q"), ("trade_date", "ts_code", "question")),
    "wc_list": (("id",),),
    "wc_cnt": (("sn",), ("account", "publish_time", "title")),
    "ths_index": (("ts_code",),),
    "ths_member": (("ts_code", "con_code"),),
    "dc_index": (("ts_code", "trade_date"), ("ts_code",)),
    "dc_member": (("ts_code", "con_code", "trade_date"),),
    "hm_list": (("name",),),
    "hm_detail": (("trade_date", "ts_code", "hm_name"),),
    "ths_hot": (("trade_date", "ts_code", "market"),),
    "dc_hot": (("trade_date", "ts_code", "market", "hot_type"),),
    "tdx_index": (("ts_code", "trade_date", "idx_type"),),
    "tdx_member": (("ts_code", "con_code", "trade_date"),),
    "kpl_list": (("trade_date", "ts_code", "tag"),),
    "kpl_concept_cons": (("trade_date", "ts_code", "con_code"),),
    "dc_concept": (("trade_date", "theme_code"),),
    "dc_concept_cons": (("trade_date", "theme_code", "ts_code"),),
    "sw_mins": (("ts_code", "trade_time"),),
    "hk_mins": (("ts_code", "trade_time"),),
    "wz_index": (("date",),),
    "gz_index": (("date",),),
}


def coverage_primary_key_candidates(dataset: str) -> tuple[tuple[str, ...], ...]:
    if dataset not in COVERAGE_DATASETS:
        return ()
    return _PRIMARY_KEY_OVERRIDES.get(
        dataset,
        (
            ("ts_code", "trade_date", "type"),
            ("ts_code", "trade_date"),
            ("trade_date", "ts_code"),
            ("ts_code", "ann_date"),
            ("ts_code", "end_date"),
            ("trade_date", "name"),
            ("date", "ts_code"),
            ("date", "name"),
            ("ts_code",),
            ("id",),
        ),
    )


def coverage_bundle_datasets(bundle: str) -> set[str]:
    if bundle not in COVERAGE_BUNDLES:
        raise ValueError(f"unsupported coverage bundle: {bundle}")
    return {rule.dataset for rule in _RULES[bundle]} | SECONDARY_DATASETS.get(bundle, set())


def coverage_specs(
    bundle: str,
    *,
    start: date,
    end: date,
    trading_dates: Iterable[str],
    max_attempts: int,
) -> list[FetchSpec]:
    if bundle not in COVERAGE_BUNDLES:
        raise ValueError(f"unsupported coverage bundle: {bundle}")
    dates = sorted(set(trading_dates)) or _weekdays(start, end)
    calendar_dates = _calendar_dates(start, end)
    specs: list[FetchSpec] = []
    for rule in _RULES[bundle]:
        variants = [dict(items) for items in rule.variants]
        if rule.mode == "once":
            for variant in variants:
                specs.extend(_paged(rule, variant, f"{rule.dataset}:{_tag(variant)}", max_attempts))
        elif rule.mode in {"daily", "calendar_daily"}:
            source_dates = dates if rule.mode == "daily" else calendar_dates
            for value in source_dates:
                for variant in variants:
                    params = {rule.date_param: value, **variant}
                    specs.extend(
                        _paged(
                            rule,
                            params,
                            f"{rule.dataset}:{value}:{_tag(variant)}",
                            max_attempts,
                            expected_date=value,
                        )
                    )
        elif rule.mode == "calendar_daily_range":
            for value in calendar_dates:
                params = {"start_date": value, "end_date": value}
                specs.extend(_paged(rule, params, f"{rule.dataset}:{value}", max_attempts))
        elif rule.mode == "month":
            for month_start, _ in _month_ranges(start, end):
                value = month_start.strftime("%Y%m")
                specs.extend(
                    _paged(
                        rule,
                        {rule.date_param: value},
                        f"{rule.dataset}:{value}",
                        max_attempts,
                    )
                )
        elif rule.mode == "month_range":
            for window_start, window_end in _month_ranges(start, end):
                params = {
                    "start_date": compact_date(window_start),
                    "end_date": compact_date(window_end),
                }
                specs.extend(
                    _paged(
                        rule,
                        params,
                        f"{rule.dataset}:{params['start_date']}:{params['end_date']}",
                        max_attempts,
                        partition=(
                            partition_metadata(
                                "date",
                                window_start,
                                window_end,
                                values=[
                                    value
                                    for value in dates
                                    if params["start_date"] <= value <= params["end_date"]
                                ],
                            )
                            if rule.dataset in _ADAPTIVE_CAPITAL_FLOW_DATASETS
                            else None
                        ),
                    )
                )
        elif rule.mode in {"year", "year_week"}:
            for window_start, window_end in _year_ranges(start, end):
                if rule.mode == "year_week":
                    params = {
                        "start_week": compact_date(window_start),
                        "end_week": compact_date(window_end),
                    }
                else:
                    params = {
                        "start_date": compact_date(window_start),
                        "end_date": compact_date(window_end),
                    }
                specs.extend(
                    _paged(
                        rule,
                        params,
                        f"{rule.dataset}:{window_start.year}",
                        max_attempts,
                        partition=(
                            partition_metadata(
                                "date",
                                window_start,
                                window_end,
                                values=[
                                    value
                                    for value in dates
                                    if compact_date(window_start)
                                    <= value
                                    <= compact_date(window_end)
                                ],
                            )
                            if rule.mode == "year"
                            and rule.dataset in _ADAPTIVE_CAPITAL_FLOW_DATASETS
                            else None
                        ),
                    )
                )
        else:
            raise ValueError(f"unsupported coverage mode: {rule.mode}")
    return specs


def coverage_secondary_specs(
    bundle: str,
    symbols_by_dataset: dict[str, Iterable[str]],
    *,
    start: date,
    end: date,
    max_attempts: int,
) -> list[FetchSpec]:
    if bundle == "cn_governance_risk":
        symbols = _codes(symbols_by_dataset.get("stk_rewards", ()))
        specs: list[FetchSpec] = []
        for batch in _batched(symbols, 100):
            params = {"ts_code": ",".join(batch)}
            specs.append(
                FetchSpec(
                    dataset="stk_rewards",
                    api_name="stk_rewards",
                    params=params,
                    scope={**params, "row_limit": 10_000},
                    allow_empty=True,
                    max_attempts=max_attempts,
                )
            )
        return apply_reference_refresh(specs, as_of=end)
    if bundle == "strategy_specialty_minutes":
        specs = []
        for dataset, api_name, freq, row_limit in (
            ("sw_mins", "sw_mins", "5min", 5_000),
            ("hk_mins", "hk_mins", "5min", 8_000),
        ):
            for symbol in _codes(symbols_by_dataset.get(dataset, ())):
                for window_start, window_end in _month_ranges(start, end):
                    params = {
                        "ts_code": symbol,
                        "freq": freq,
                        "start_date": f"{window_start.isoformat()} 00:00:00",
                        "end_date": f"{window_end.isoformat()} 23:59:59",
                    }
                    specs.append(
                        FetchSpec(
                            dataset=dataset,
                            api_name=api_name,
                            params=params,
                            scope={**params, "row_limit": row_limit},
                            allow_empty=True,
                            max_attempts=max_attempts,
                        )
                    )
        return specs
    return []


def _paged(
    rule: CoverageRule,
    base_params: dict[str, Any],
    group: str,
    max_attempts: int,
    *,
    expected_date: str | None = None,
    partition: dict[str, Any] | None = None,
) -> list[FetchSpec]:
    scope: dict[str, Any] = {
        **base_params,
        "page_group": group.rstrip(":"),
        "page_size": rule.page_size,
        "max_pages": rule.max_pages,
        "offset": 0,
        **(partition or {}),
    }
    if expected_date:
        scope.update(
            {
                "expected_date_field": rule.date_field,
                "expected_date": expected_date,
            }
        )
    elif partition:
        scope.update(
            {
                "expected_date_field": rule.date_field,
                "expected_date_start": str(partition["partition_start"])
                .replace("-", "")[:8],
                "expected_date_end": str(partition["partition_end"])
                .replace("-", "")[:8],
            }
        )
    return [
        FetchSpec(
            dataset=rule.dataset,
            api_name=rule.dataset,
            params={**base_params, "limit": rule.page_size, "offset": 0},
            scope=scope,
            allow_empty=True,
            max_attempts=max_attempts,
        )
    ]


def _tag(values: dict[str, Any]) -> str:
    return ":".join(f"{key}={values[key]}" for key in sorted(values))


def _codes(values: Iterable[str]) -> list[str]:
    return sorted({str(value).strip() for value in values if str(value).strip()})


def _batched(values: list[str], size: int) -> Iterable[list[str]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def _weekdays(start: date, end: date) -> list[str]:
    values = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            values.append(compact_date(current))
        current += timedelta(days=1)
    return values


def _calendar_dates(start: date, end: date) -> list[str]:
    values = []
    current = start
    while current <= end:
        values.append(compact_date(current))
        current += timedelta(days=1)
    return values


def _month_ranges(start: date, end: date) -> list[tuple[date, date]]:
    values = []
    current = start
    while current <= end:
        next_month = (
            date(current.year + 1, 1, 1)
            if current.month == 12
            else date(current.year, current.month + 1, 1)
        )
        window_end = min(end, next_month - timedelta(days=1))
        values.append((current, window_end))
        current = window_end + timedelta(days=1)
    return values


def _year_ranges(start: date, end: date) -> list[tuple[date, date]]:
    values = []
    current = start
    while current <= end:
        window_end = min(end, date(current.year, 12, 31))
        values.append((current, window_end))
        current = window_end + timedelta(days=1)
    return values
