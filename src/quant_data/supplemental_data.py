from __future__ import annotations

from calendar import monthrange
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import date, timedelta
from typing import Any

from .coverage_data import COVERAGE_BUNDLES, coverage_bundle_datasets, coverage_specs
from .history_bounds import clip_history_range, history_start_date
from .models import FetchSpec, ProviderResult
from .partitioning import is_adaptive_partition, partition_metadata, split_partition_spec
from .planner import compact_date
from .provider import ProviderError
from .reference_data import apply_reference_refresh

SUPPORTED_BUNDLES = {
    "cn_extended_daily",
    "cn_funds",
    "cn_macro",
    "cn_institutional",
    "cn_futures",
    "cn_options_bonds",
    "hk_market",
    "us_market",
    "global_markets",
} | COVERAGE_BUNDLES

_CN_EXCHANGES = ("CFFEX", "DCE", "CZCE", "SHFE", "INE", "GFEX")
ETF_CONSTITUENT_DATASETS = {"etf_sh_cons", "etf_sz_cons"}
SHARE_FLOAT_PROVIDER_OFFSET_CAP = 100_000

# ``fina_indicator_vip`` omits several documented research fields when the
# request leaves ``fields`` empty.  Keep an explicit, versioned field contract
# so a provider-default change cannot silently remove factors from future
# snapshots.  The list contains every column observed in the legacy default
# response plus the five non-default columns consumed by Qlib.
FINA_INDICATOR_EXPLICIT_FIELDS = (
    "ts_code",
    "ann_date",
    "end_date",
    "eps",
    "dt_eps",
    "total_revenue_ps",
    "revenue_ps",
    "capital_rese_ps",
    "surplus_rese_ps",
    "undist_profit_ps",
    "extra_item",
    "profit_dedt",
    "gross_margin",
    "current_ratio",
    "quick_ratio",
    "cash_ratio",
    "inv_turn",
    "ar_turn",
    "ca_turn",
    "fa_turn",
    "assets_turn",
    "op_income",
    "ebit",
    "ebitda",
    "fcff",
    "fcfe",
    "current_exint",
    "noncurrent_exint",
    "interestdebt",
    "netdebt",
    "tangible_asset",
    "working_capital",
    "networking_capital",
    "invest_capital",
    "retained_earnings",
    "diluted2_eps",
    "bps",
    "ocfps",
    "retainedps",
    "cfps",
    "ebit_ps",
    "fcff_ps",
    "fcfe_ps",
    "netprofit_margin",
    "grossprofit_margin",
    "cogs_of_sales",
    "expense_of_sales",
    "profit_to_gr",
    "saleexp_to_gr",
    "adminexp_of_gr",
    "finaexp_of_gr",
    "impai_ttm",
    "gc_of_gr",
    "op_of_gr",
    "ebit_of_gr",
    "roe",
    "roe_waa",
    "roe_dt",
    "roa",
    "npta",
    "roic",
    "roe_yearly",
    "roa2_yearly",
    "salescash_to_or",
    "ocf_to_or",
    "ocf_to_profit",
    "debt_to_assets",
    "assets_to_eqt",
    "dp_assets_to_eqt",
    "ca_to_assets",
    "nca_to_assets",
    "tbassets_to_totalassets",
    "int_to_talcap",
    "eqt_to_talcapital",
    "currentdebt_to_debt",
    "longdeb_to_debt",
    "ocf_to_shortdebt",
    "debt_to_eqt",
    "eqt_to_debt",
    "eqt_to_interestdebt",
    "tangibleasset_to_debt",
    "tangasset_to_intdebt",
    "tangibleasset_to_netdebt",
    "ocf_to_debt",
    "turn_days",
    "roa_yearly",
    "roa_dp",
    "fixed_assets",
    "profit_to_op",
    "q_saleexp_to_gr",
    "q_gc_to_gr",
    "q_roe",
    "q_dt_roe",
    "q_npta",
    "q_ocf_to_sales",
    "q_profit_yoy",
    "basic_eps_yoy",
    "dt_eps_yoy",
    "cfps_yoy",
    "op_yoy",
    "ebt_yoy",
    "netprofit_yoy",
    "dt_netprofit_yoy",
    "ocf_yoy",
    "roe_yoy",
    "bps_yoy",
    "assets_yoy",
    "eqt_yoy",
    "tr_yoy",
    "or_yoy",
    "q_sales_yoy",
    "q_op_qoq",
    "equity_yoy",
    "update_flag",
)
FINA_INDICATOR_FIELD_CONTRACT = "fina-indicator-explicit-fields-v1"
_PAGINATION_MAX_PAGES = {
    "index_basic": 16,
    # These are safety ceilings, not pre-planned page counts.  The CLI starts
    # with one page and stops as soon as Tushare returns a short page.  Dense
    # month/day partitions observed in production legitimately exceed the old
    # 8/16 page ceilings, so keep the ceiling high enough to prove termination
    # without silently accepting a truncated partition.
    "stk_holdernumber": 64,
    "top10_holders": 64,
    "top10_floatholders": 64,
    "stock_st": 4,
    "sw_daily": 4,
    "income": 64,
    "balancesheet": 64,
    "cashflow": 64,
    "fina_indicator": 64,
    "forecast": 64,
    "express": 64,
    "namechange": 64,
    "dividend": 16,
    "repurchase": 64,
    # A single monthly unlock-date partition exceeded 64,000 holder-level
    # rows in production.  Keep the existing 1,000-row page keys so completed
    # pages remain reusable, but permit pagination to continue until a short
    # terminal page proves completeness.
    "share_float": 512,
    "pledge_stat": 16,
    "pledge_detail": 16,
    "stk_holdertrade": 64,
    "anns_d": 64,
    "block_trade": 4,
    "hk_hold": 8,
    "stk_factor_pro": 4,
    "fina_audit": 16,
    "new_share": 4,
    "fina_mainbz": 64,
    "fund_basic": 16,
    "fund_company": 4,
    "fund_manager": 64,
    "fund_nav": 64,
    "fund_share": 64,
    "fund_div": 32,
    "fund_portfolio": 1_024,
    "etf_basic": 4,
    "etf_index": 4,
    "fut_mapping": 8,
    "fut_daily": 4,
    "fut_holding": 64,
    "fut_wsr": 4,
    "fut_settle": 4,
    "ft_limit": 4,
    # opt_basic is an all-history contract master. Production exceeded the
    # former 64-page ceiling (128k rows), so allow the short-page proof to
    # continue without accepting a truncated catalog.
    "opt_basic": 1_024,
    "cb_basic": 6,
    "cb_issue": 8,
    "cb_redeem": 8,
    "cb_rate": 8,
    "cb_price_chg": 8,
    "cb_share": 8,
    "opt_daily": 64,
    "cb_daily": 4,
    "repo_daily": 4,
    "yc_cb": 4,
    "hk_basic": 8,
    "hk_daily": 8,
    "hk_daily_adj": 8,
    "us_basic": 30,
    "us_daily": 16,
    "us_daily_adj": 16,
    # US VIP statement cross-sections are much denser than their A-share
    # equivalents. Production probes for a single report period still
    # returned full 1,000-row pages beyond offset 128k, while all three
    # endpoints terminated before offset 256k. Keep a generous safety ceiling
    # so completeness is proved by a short page instead of silently accepting
    # the former 64-page truncation.
    "us_income": 512,
    "us_balancesheet": 512,
    "us_cashflow": 512,
    "us_fina_indicator": 64,
    "fx_obasic": 8,
    "index_global": 8,
    "fx_daily": 8,
    "report_rc": 64,
    "etf_sh_cons": 256,
    "etf_sz_cons": 256,
    "ci_daily": 4,
    "major_news": 512,
}

_PAGINATION_PREFETCH_PAGES = {
    # CSI currently spans nine 1,000-row pages. Fetch a bounded window while
    # retaining the mandatory short-page termination proof.
    "index_basic": 8,
    # Contract-master offsets are independent. Fetch a small bounded window in
    # parallel so one full page does not force another complete CLI phase.
    "opt_basic": 8,
    # Report-period pages are independent. A bounded window avoids hundreds
    # of planner phases while the shared provider rate gate remains the hard
    # request ceiling.
    "us_income": 8,
    "us_balancesheet": 8,
    "us_cashflow": 8,
}


def supplemental_specs(
    bundle: str,
    *,
    start: date,
    end: date,
    trading_dates: Iterable[str],
    max_attempts: int,
) -> list[FetchSpec]:
    if bundle not in SUPPORTED_BUNDLES:
        raise ValueError(f"unsupported supplemental bundle: {bundle}")
    if bundle in COVERAGE_BUNDLES:
        specs = coverage_specs(
            bundle,
            start=start,
            end=end,
            trading_dates=trading_dates,
            max_attempts=max_attempts,
        )
    else:
        dates = sorted(set(trading_dates))
        if bundle == "cn_extended_daily":
            specs = _cn_extended_specs(start, end, dates, max_attempts)
        elif bundle == "cn_funds":
            specs = _cn_fund_specs(start, end, dates, max_attempts)
        elif bundle == "cn_macro":
            specs = _cn_macro_specs(start, end, max_attempts)
        elif bundle == "cn_institutional":
            specs = _cn_institutional_specs(start, end, dates, max_attempts)
        elif bundle == "cn_futures":
            specs = _cn_futures_specs(start, end, dates, max_attempts)
        elif bundle == "cn_options_bonds":
            specs = _cn_options_bonds_specs(dates, max_attempts)
        elif bundle == "hk_market":
            specs = _hk_market_specs(start, end, max_attempts)
        elif bundle == "us_market":
            specs = _us_market_specs(start, end, max_attempts)
        else:
            specs = _global_market_specs(start, end, max_attempts)
    return apply_reference_refresh(specs, as_of=end)


def validate_supplemental(spec: FetchSpec, result: ProviderResult) -> ProviderResult:
    if spec.dataset == "bc_otcqt":
        # This endpoint uniquely returns upper-case provider fields, unlike the
        # rest of Tushare.  Normalize at the adapter boundary so date contracts,
        # primary keys, and downstream factor code use one canonical schema.
        result = ProviderResult(
            api_name=result.api_name,
            columns=[str(column).lower() for column in result.columns],
            rows=[{str(key).lower(): value for key, value in row.items()} for row in result.rows],
            raw_body=result.raw_body,
            metadata=result.metadata,
        )
    page_size = spec.scope.get("page_size")
    if page_size is not None and len(result.rows) > int(page_size):
        raise ProviderError(
            f"{spec.dataset} ignored the requested page size {page_size}: "
            f"returned {len(result.rows)} rows",
            retryable=False,
        )
    row_limit = spec.scope.get("row_limit")
    if row_limit is not None and len(result.rows) >= int(row_limit):
        raise ProviderError(
            f"{spec.dataset} returned {len(result.rows)} rows and may be truncated at "
            f"the {row_limit}-row provider limit; use a smaller time partition",
            retryable=False,
        )
    expected_field = spec.scope.get("expected_date_field")
    expected_date = spec.scope.get("expected_date")
    if expected_field and expected_date:
        for row in result.rows:
            actual = str(row.get(str(expected_field)) or "").replace("-", "")[:8]
            if actual != expected_date:
                raise ProviderError(
                    f"{spec.dataset} returned {expected_field}={actual!r}, "
                    f"expected {expected_date}",
                    retryable=False,
                )
    expected_start = spec.scope.get("expected_date_start")
    expected_end = spec.scope.get("expected_date_end")
    if expected_field and expected_start and expected_end:
        for row in result.rows:
            actual = str(row.get(str(expected_field)) or "").replace("-", "")[:8]
            if not str(expected_start) <= actual <= str(expected_end):
                raise ProviderError(
                    f"{spec.dataset} returned {expected_field}={actual!r} outside "
                    f"{expected_start}..{expected_end}",
                    retryable=False,
                )
    return result


def require_pagination_terminated(
    specs: Iterable[FetchSpec], successful_rows: Iterable[dict[str, Any]]
) -> None:
    row_counts = {str(row["unit_key"]): int(row.get("row_count") or 0) for row in successful_rows}
    groups: dict[str, list[FetchSpec]] = defaultdict(list)
    for spec in specs:
        group = spec.scope.get("page_group")
        if group:
            groups[str(group)].append(spec)
    continued_groups = {parent for spec in specs for parent in _linked_page_groups(spec.scope)}
    unterminated: list[str] = []
    for group, pages in groups.items():
        if group in continued_groups:
            continue
        ordered = sorted(pages, key=lambda item: int(item.scope["offset"]))
        counts = [row_counts.get(page.unit_key, -1) for page in ordered]
        terminated = bool(counts) and 0 <= counts[-1] < int(ordered[-1].scope["page_size"])
        if not terminated:
            unterminated.append(group)
    if unterminated:
        preview = ", ".join(unterminated[:5])
        raise RuntimeError(
            f"pagination did not reach a short terminal page for {len(unterminated)} "
            f"partitions: {preview}; increase max_pages before accepting this download"
        )


def next_pagination_specs(
    specs: Iterable[FetchSpec], successful_rows: Iterable[dict[str, Any]]
) -> list[FetchSpec]:
    row_counts = {str(row["unit_key"]): int(row.get("row_count") or 0) for row in successful_rows}
    groups: dict[str, list[FetchSpec]] = defaultdict(list)
    for spec in specs:
        group = spec.scope.get("page_group")
        if group:
            groups[str(group)].append(spec)
    continued_groups = {parent for spec in specs for parent in _linked_page_groups(spec.scope)}
    result: list[FetchSpec] = []
    for group, pages in groups.items():
        if group in continued_groups:
            continue
        current = max(pages, key=lambda item: int(item.scope["offset"]))
        page_size = int(current.scope["page_size"])
        if row_counts.get(current.unit_key, -1) < page_size:
            continue
        current_page = int(
            current.scope.get("page_index", int(current.scope["offset"]) // page_size)
        )
        next_page = current_page + 1
        if (
            current.dataset == "share_float"
            and next_page * page_size >= SHARE_FLOAT_PROVIDER_OFFSET_CAP
        ):
            # Tushare rejects this cursor deterministically.  Let the caller
            # replace the full page group with disjoint date/symbol
            # partitions instead of sending an invalid request and triggering
            # the provider's multi-minute cooldown.
            continue
        max_pages = int(
            current.scope.get("max_pages")
            or _PAGINATION_MAX_PAGES[current.dataset]
        )
        if next_page >= max_pages:
            continue
        prefetch = _PAGINATION_PREFETCH_PAGES.get(current.dataset, 1)
        for page in range(next_page, min(max_pages, next_page + prefetch)):
            result.append(_next_page_spec(current, page))
    return result


def pagination_extension_spec(current: FetchSpec, *, max_pages: int) -> FetchSpec:
    """Continue a capped page group without changing its completed unit keys.

    ``max_pages`` is the safety ceiling for the continuation itself.  The new
    group starts at the first offset after ``current`` and records the parent
    group so pagination verification treats the original full page as safely
    continued rather than truncated.
    """

    if max_pages < 1:
        raise ValueError("pagination continuation max_pages must be positive")
    page_size = int(current.scope["page_size"])
    next_offset = int(current.scope.get("offset") or 0) + page_size
    parent_group = str(current.scope["page_group"])
    continuation_group = f"{parent_group}:continuation:{next_offset}"
    params = {
        **current.params,
        "limit": page_size,
        "offset": next_offset,
    }
    scope = {
        **current.scope,
        "page_group": continuation_group,
        "continues_page_group": parent_group,
        "offset_origin": next_offset,
        "offset": next_offset,
        "page_index": 0,
        "max_pages": max_pages,
    }
    return _spec(
        current.dataset,
        current.api_name,
        params,
        scope=scope,
        fields=current.fields,
        allow_empty=current.allow_empty,
        max_attempts=current.max_attempts,
    )


def _linked_page_groups(scope: Mapping[str, Any]) -> set[str]:
    groups = {
        str(value)
        for key in ("continues_page_group", "supersedes_page_group")
        if (value := scope.get(key))
    }
    plural = scope.get("supersedes_page_groups")
    if isinstance(plural, (list, tuple, set, frozenset)):
        groups.update(str(value) for value in plural if value)
    return groups


def tdx_member_overflow_repartition_specs(
    failed_spec: FetchSpec,
    symbols: Iterable[str],
) -> list[FetchSpec]:
    """Replace an offset-capped TDX member day with per-index partitions."""

    if failed_spec.dataset != "tdx_member":
        raise ValueError("overflow repartition is only supported for tdx_member")
    trade_date = str(failed_spec.params.get("trade_date") or "")
    if len(trade_date) != 8:
        raise ValueError("tdx_member overflow requires a compact trade_date")
    normalized_symbols = sorted(
        {str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()}
    )
    if not normalized_symbols:
        raise RuntimeError(
            f"tdx_member partition {trade_date} exceeded the provider offset cap "
            "and tdx_index supplied no index symbols"
        )

    current_group = str(failed_spec.scope["page_group"])
    root_group = str(failed_spec.scope.get("continues_page_group") or current_group)
    superseded_groups = sorted({current_group, root_group})
    page_size = int(failed_spec.scope.get("page_size") or 3_000)
    max_pages = int(failed_spec.scope.get("max_pages") or 8)
    return [
        _spec(
            "tdx_member",
            "tdx_member",
            {
                "trade_date": trade_date,
                "ts_code": symbol,
                "limit": page_size,
                "offset": 0,
            },
            scope={
                "trade_date": trade_date,
                "ts_code": symbol,
                "page_group": f"{root_group}:symbol:{symbol}",
                "offset": 0,
                "page_size": page_size,
                "max_pages": max_pages,
                "expected_date_field": "trade_date",
                "expected_date": trade_date,
                "supersedes_page_groups": superseded_groups,
            },
            allow_empty=True,
            max_attempts=failed_spec.max_attempts,
        )
        for symbol in normalized_symbols
    ]


def share_float_overflow_repartition_specs(
    failed_spec: FetchSpec,
    symbols: Iterable[str] = (),
) -> list[FetchSpec]:
    """Replace an offset-capped share-float window with disjoint partitions.

    Provider probes proved that an exact-day query does not preserve the row
    order of the enclosing monthly query, so a cursor cannot safely resume at
    the monthly boundary. A multi-day window is therefore replaced by daily
    groups. If one exact day itself exceeds the provider offset cap, the
    documented ``ts_code`` filter becomes the final disjoint partition axis.
    Successful parent units remain immutable while the replacement generation
    explicitly supersedes that page group in successor snapshots.
    """

    if failed_spec.dataset != "share_float":
        raise ValueError("overflow continuation is only supported for share_float")
    start_text = str(failed_spec.params.get("start_date") or "")
    end_text = str(failed_spec.params.get("end_date") or "")
    if len(start_text) != 8 or len(end_text) != 8:
        raise ValueError("share_float overflow requires compact date bounds")
    window_start = date.fromisoformat(
        f"{start_text[:4]}-{start_text[4:6]}-{start_text[6:8]}"
    )
    window_end = date.fromisoformat(
        f"{end_text[:4]}-{end_text[4:6]}-{end_text[6:8]}"
    )
    parent_group = str(failed_spec.scope["page_group"])
    page_size = 6_000
    if window_start == window_end:
        normalized_symbols = sorted(
            {
                str(symbol).strip().upper()
                for symbol in symbols
                if str(symbol).strip().upper().endswith((".SH", ".SZ", ".BJ"))
            }
        )
        if not normalized_symbols:
            raise RuntimeError(
                f"single-day share_float partition {start_text} exceeded the "
                "provider offset cap and stock_basic supplied no active symbols"
            )
        return [
            _spec(
                "share_float",
                "share_float",
                {
                    "ts_code": symbol,
                    "start_date": start_text,
                    "end_date": end_text,
                    "limit": page_size,
                    "offset": 0,
                },
                scope={
                    "ts_code": symbol,
                    "start_date": start_text,
                    "end_date": end_text,
                    "page_group": f"{parent_group}:symbol:{symbol}",
                    "offset": 0,
                    "page_size": page_size,
                    "max_pages": _PAGINATION_MAX_PAGES["share_float"],
                    "expected_date_field": "float_date",
                    "expected_date": start_text,
                    "supersedes_page_group": parent_group,
                },
                allow_empty=True,
                max_attempts=failed_spec.max_attempts,
            )
            for symbol in normalized_symbols
        ]

    result: list[FetchSpec] = []
    current = window_start
    while current <= window_end:
        value = compact_date(current)
        result.append(
            _spec(
                "share_float",
                "share_float",
                {
                    "start_date": value,
                    "end_date": value,
                    "limit": page_size,
                    "offset": 0,
                },
                scope={
                    "start_date": value,
                    "end_date": value,
                    "page_group": f"{parent_group}:daily:{value}",
                    "offset": 0,
                    "page_size": page_size,
                    "max_pages": _PAGINATION_MAX_PAGES["share_float"],
                    "expected_date_field": "float_date",
                    "expected_date": value,
                    "supersedes_page_group": parent_group,
                },
                allow_empty=True,
                max_attempts=failed_spec.max_attempts,
            )
        )
        current += timedelta(days=1)
    return result


def etf_constituent_history_specs(
    active_ranges: Mapping[str, tuple[date, date]], *, max_attempts: int
) -> list[FetchSpec]:
    """Plan ETF constituent history by symbol and the largest safe date window.

    Full-market date partitions exceed the provider's 100,000-row offset cap
    and previously expanded into one request per ETF per trading day.  A
    symbol/date-range partition keeps the same rows while allowing the normal
    pagination loop to cover many trading days per request series.
    """

    specs: list[FetchSpec] = []
    for symbol, (window_start, window_end) in sorted(active_ranges.items()):
        normalized = str(symbol).strip().upper()
        if normalized.endswith(".SH"):
            dataset = "etf_sh_cons"
        elif normalized.endswith(".SZ"):
            dataset = "etf_sz_cons"
        else:
            continue
        if window_end < window_start:
            continue
        start_text = compact_date(window_start)
        end_text = compact_date(window_end)
        specs.extend(
            _paged_specs(
                dataset,
                dataset,
                {
                    "ts_code": normalized,
                    "start_date": start_text,
                    "end_date": end_text,
                },
                group=f"{dataset}:{normalized}:{start_text}:{end_text}",
                page_size=3_000,
                max_pages=_PAGINATION_MAX_PAGES[dataset],
                max_attempts=max_attempts,
                expected_date_field="trade_date",
                partition=partition_metadata("date", window_start, window_end),
            )
        )
    return specs


def etf_constituent_overflow_repartition_specs(
    failed_spec: FetchSpec, symbols: Iterable[str] = ()
) -> list[FetchSpec]:
    """Replace an offset-capped ETF partition with a smaller documented grain.

    New history plans are already partitioned by ETF symbol, so an oversized
    range is bisected by date.  The legacy full-market/date plan remains
    recoverable by symbol for old checkpoints.  Existing successful pages stay
    immutable; replacement groups supersede the capped parent group.
    """

    if failed_spec.dataset not in ETF_CONSTITUENT_DATASETS:
        raise ValueError("ETF overflow continuation requires an ETF constituent dataset")
    offset = int(failed_spec.params.get("offset") or 0)
    if offset < 100_000:
        raise ValueError("ETF overflow continuation requires an offset-capped page")

    symbol = str(failed_spec.params.get("ts_code") or "").strip().upper()
    start_text = str(failed_spec.params.get("start_date") or "")
    end_text = str(failed_spec.params.get("end_date") or "")
    if symbol and len(start_text) == 8 and len(end_text) == 8:
        if is_adaptive_partition(failed_spec):
            return split_partition_spec(failed_spec)
        window_start = date.fromisoformat(
            f"{start_text[:4]}-{start_text[4:6]}-{start_text[6:8]}"
        )
        window_end = date.fromisoformat(
            f"{end_text[:4]}-{end_text[4:6]}-{end_text[6:8]}"
        )
        if window_end <= window_start:
            raise RuntimeError(
                f"single-day ETF constituent partition {symbol}/{start_text} "
                "exceeded the provider offset cap"
            )
        midpoint = window_start + timedelta(days=(window_end - window_start).days // 2)
        parent_group = str(failed_spec.scope["page_group"])
        result: list[FetchSpec] = []
        for child_start, child_end in (
            (window_start, midpoint),
            (midpoint + timedelta(days=1), window_end),
        ):
            child_start_text = compact_date(child_start)
            child_end_text = compact_date(child_end)
            group = (
                f"{failed_spec.dataset}:{symbol}:"
                f"{child_start_text}:{child_end_text}"
            )
            result.append(
                _spec(
                    failed_spec.dataset,
                    failed_spec.api_name,
                    {
                        "ts_code": symbol,
                        "start_date": child_start_text,
                        "end_date": child_end_text,
                        "limit": 3_000,
                        "offset": 0,
                    },
                    scope={
                        "ts_code": symbol,
                        "start_date": child_start_text,
                        "end_date": child_end_text,
                        "page_group": group,
                        "offset": 0,
                        "page_size": 3_000,
                        "max_pages": _PAGINATION_MAX_PAGES[failed_spec.dataset],
                        "expected_date_field": "trade_date",
                        "expected_date_start": child_start_text,
                        "expected_date_end": child_end_text,
                        "supersedes_page_group": parent_group,
                    },
                    allow_empty=True,
                    max_attempts=failed_spec.max_attempts,
                )
            )
        return result

    trade_date = str(failed_spec.params.get("trade_date") or "")
    if len(trade_date) != 8:
        raise ValueError(
            "ETF overflow continuation requires a compact date range or trade_date"
        )

    suffix = ".SH" if failed_spec.dataset == "etf_sh_cons" else ".SZ"
    eligible = sorted(
        {
            str(symbol).strip().upper()
            for symbol in symbols
            if str(symbol).strip().upper().endswith(suffix)
        }
    )
    if not eligible:
        raise RuntimeError(
            f"no {suffix} ETF symbols are available for {failed_spec.dataset} recovery"
        )

    parent_group = str(failed_spec.scope["page_group"])
    page_size = 3_000
    result: list[FetchSpec] = []
    for symbol in eligible:
        group = f"{parent_group}:symbol:{symbol}"
        result.append(
            _spec(
                failed_spec.dataset,
                failed_spec.api_name,
                {
                    "ts_code": symbol,
                    "trade_date": trade_date,
                    "limit": page_size,
                    "offset": 0,
                },
                scope={
                    "ts_code": symbol,
                    "trade_date": trade_date,
                    "page_group": group,
                    "offset": 0,
                    "page_size": page_size,
                    "max_pages": _PAGINATION_MAX_PAGES[failed_spec.dataset],
                    "expected_date_field": "trade_date",
                    "expected_date": trade_date,
                    "supersedes_page_group": parent_group,
                },
                allow_empty=True,
                max_attempts=failed_spec.max_attempts,
            )
        )
    return result


def bundle_datasets(bundle: str) -> set[str]:
    if bundle in COVERAGE_BUNDLES:
        return coverage_bundle_datasets(bundle)
    marker = date(2024, 1, 2)
    datasets = {
        spec.dataset
        for spec in supplemental_specs(
            bundle,
            start=marker,
            end=marker,
            trading_dates=[compact_date(marker)],
            max_attempts=1,
        )
    }
    if bundle in {"hk_market", "us_market"}:
        market = bundle.split("_", 1)[0]
        datasets.update(
            {
                f"{market}_daily",
                f"{market}_daily_adj",
                f"{market}_income",
                f"{market}_balancesheet",
                f"{market}_cashflow",
                f"{market}_fina_indicator",
            }
        )
    if bundle == "cn_options_bonds":
        datasets.update(
            {
                "cb_rate",
                "cb_price_chg",
                "cb_share",
                "cb_rating",
                "top10_cb_holders",
            }
        )
    if bundle == "cn_institutional":
        datasets.update(ETF_CONSTITUENT_DATASETS)
    return datasets


def a_share_bulk_history_specs(*, start: date, end: date, max_attempts: int) -> list[FetchSpec]:
    """Plan full-market financial and event history without per-stock requests."""
    specs: list[FetchSpec] = []
    for period in _financial_periods(start, end):
        for dataset, api_name in (
            ("income", "income_vip"),
            ("balancesheet", "balancesheet_vip"),
            ("cashflow", "cashflow_vip"),
            ("fina_indicator", "fina_indicator_vip"),
            ("forecast", "forecast_vip"),
            ("express", "express_vip"),
        ):
            legacy_page_group = f"{dataset}:{period}"
            page_group = legacy_page_group
            fields: tuple[str, ...] = ()
            partition: dict[str, Any] | None = None
            if dataset == "fina_indicator":
                page_group = f"{legacy_page_group}:{FINA_INDICATOR_FIELD_CONTRACT}"
                fields = FINA_INDICATOR_EXPLICIT_FIELDS
                partition = {
                    "field_contract": FINA_INDICATOR_FIELD_CONTRACT,
                    "supersedes_page_group": legacy_page_group,
                }
            specs.extend(
                _paged_specs(
                    dataset,
                    api_name,
                    {"period": period},
                    group=page_group,
                    page_size=1_000,
                    max_pages=_PAGINATION_MAX_PAGES[dataset],
                    max_attempts=max_attempts,
                    fields=fields,
                    expected_date_field="end_date",
                    expected_date=period,
                    partition=partition,
                )
            )

    for dataset in ("namechange", "repurchase", "share_float", "stk_holdertrade"):
        history_range = clip_history_range(dataset, start, end)
        for window_start, window_end in (
            _month_ranges(*history_range) if history_range is not None else ()
        ):
            compact_start = compact_date(window_start)
            compact_end = compact_date(window_end)
            params = {"start_date": compact_start, "end_date": compact_end}
            specs.extend(
                _paged_specs(
                    dataset,
                    dataset,
                    params,
                    group=f"{dataset}:{compact_start}:{compact_end}",
                    page_size=1_000,
                    max_pages=_PAGINATION_MAX_PAGES[dataset],
                    max_attempts=max_attempts,
                )
            )

    dates = _calendar_dates(start, end)
    for dataset, date_param, date_field, max_pages in (
        ("dividend", "ann_date", "ann_date", 16),
        ("pledge_stat", "end_date", "end_date", 16),
        ("pledge_detail", "ann_date", "ann_date", 16),
        ("anns_d", "ann_date", "ann_date", 64),
    ):
        lower_bound = history_start_date(dataset)
        dataset_dates = (
            [value for value in dates if value >= compact_date(lower_bound)]
            if lower_bound is not None
            else dates
        )
        specs.extend(
            _daily_paged_specs(
                dataset,
                dataset,
                dataset_dates,
                date_param=date_param,
                date_field=date_field,
                page_size=1_000,
                max_pages=max_pages,
                max_attempts=max_attempts,
            )
        )
    return specs


def _spec(
    dataset: str,
    api_name: str,
    params: dict[str, Any],
    *,
    scope: dict[str, Any] | None = None,
    fields: tuple[str, ...] = (),
    allow_empty: bool = True,
    max_attempts: int,
) -> FetchSpec:
    return FetchSpec(
        dataset=dataset,
        api_name=api_name,
        scope=dict(scope or params),
        params=params,
        fields=fields,
        allow_empty=allow_empty,
        max_attempts=max_attempts,
    )


def _paged_specs(
    dataset: str,
    api_name: str,
    base_params: dict[str, Any],
    *,
    group: str,
    page_size: int,
    max_pages: int,
    max_attempts: int,
    fields: tuple[str, ...] = (),
    expected_date_field: str | None = None,
    expected_date: str | None = None,
    partition: dict[str, Any] | None = None,
) -> list[FetchSpec]:
    configured_max = _PAGINATION_MAX_PAGES.get(dataset)
    if configured_max != max_pages:
        raise ValueError(
            f"pagination limit mismatch for {dataset}: {max_pages} != {configured_max}"
        )
    scope: dict[str, Any] = {
        **base_params,
        "page_group": group,
        "page_size": page_size,
        "offset": 0,
        **(partition or {}),
        **({"max_pages": max_pages} if partition else {}),
    }
    if expected_date_field and expected_date:
        scope.update({"expected_date_field": expected_date_field, "expected_date": expected_date})
    elif expected_date_field and partition:
        scope.update(
            {
                "expected_date_field": expected_date_field,
                "expected_date_start": str(partition["partition_start"]).replace("-", "")[:8],
                "expected_date_end": str(partition["partition_end"]).replace("-", "")[:8],
            }
        )
    return [
        _spec(
            dataset,
            api_name,
            {**base_params, "limit": page_size, "offset": 0},
            scope=scope,
            fields=fields,
            allow_empty=True,
            max_attempts=max_attempts,
        )
    ]


def _next_page_spec(current: FetchSpec, page: int) -> FetchSpec:
    page_size = int(current.scope["page_size"])
    offset_origin = int(current.scope.get("offset_origin", 0))
    offset = offset_origin + page * page_size
    params = {**current.params, "limit": page_size, "offset": offset}
    scope = {**current.scope, "offset": offset}
    if "page_index" in current.scope:
        scope["page_index"] = page
    return _spec(
        current.dataset,
        current.api_name,
        params,
        scope=scope,
        fields=current.fields,
        allow_empty=True,
        max_attempts=current.max_attempts,
    )


def _daily_paged_specs(
    dataset: str,
    api_name: str,
    dates: Iterable[str],
    *,
    date_param: str = "trade_date",
    date_field: str = "trade_date",
    page_size: int,
    max_pages: int,
    max_attempts: int,
) -> list[FetchSpec]:
    specs: list[FetchSpec] = []
    for trading_date in dates:
        specs.extend(
            _paged_specs(
                dataset,
                api_name,
                {date_param: trading_date},
                group=f"{dataset}:{trading_date}",
                page_size=page_size,
                max_pages=max_pages,
                max_attempts=max_attempts,
                expected_date_field=date_field,
                expected_date=trading_date,
            )
        )
    return specs


def _cn_extended_specs(
    start: date, end: date, dates: list[str], max_attempts: int
) -> list[FetchSpec]:
    specs: list[FetchSpec] = []
    for dataset, page_size, max_pages in (
        ("stock_st", 1_000, 4),
        ("sw_daily", 4_000, 4),
    ):
        lower_bound = history_start_date(dataset)
        dataset_dates = (
            [value for value in dates if value >= compact_date(lower_bound)]
            if lower_bound is not None
            else dates
        )
        specs.extend(
            _daily_paged_specs(
                dataset,
                dataset,
                dataset_dates,
                page_size=page_size,
                max_pages=max_pages,
                max_attempts=max_attempts,
            )
        )
    for window_start, window_end in _month_ranges(start, end):
        compact_start = compact_date(window_start)
        compact_end = compact_date(window_end)
        params = {"start_date": compact_start, "end_date": compact_end}
        for dataset in ("stk_holdernumber", "top10_holders", "top10_floatholders"):
            specs.extend(
                _paged_specs(
                    dataset,
                    dataset,
                    params,
                    group=f"{dataset}:{compact_start}:{compact_end}",
                    page_size=1_000,
                    max_pages=64,
                    max_attempts=max_attempts,
                )
            )
    for survey_date in _weekdays(start, end):
        params = {"start_date": survey_date, "end_date": survey_date}
        specs.append(
            _spec(
                "stk_surv",
                "stk_surv",
                params,
                scope={**params, "row_limit": 400},
                max_attempts=max_attempts,
            )
        )
    specs.extend(
        _daily_paged_specs(
            "block_trade",
            "block_trade",
            dates,
            page_size=500,
            max_pages=4,
            max_attempts=max_attempts,
        )
    )
    specs.extend(
        _daily_paged_specs(
            "hk_hold",
            "hk_hold",
            dates,
            page_size=1_000,
            max_pages=8,
            max_attempts=max_attempts,
        )
    )
    specs.extend(
        _daily_paged_specs(
            "stk_factor_pro",
            "stk_factor_pro",
            dates,
            page_size=2_000,
            max_pages=4,
            max_attempts=max_attempts,
        )
    )
    for window_start, window_end in _month_ranges(start, end):
        params = {
            "start_date": compact_date(window_start),
            "end_date": compact_date(window_end),
        }
        specs.extend(
            _paged_specs(
                "new_share",
                "new_share",
                params,
                group=f"new_share:{params['start_date']}:{params['end_date']}",
                page_size=1_000,
                max_pages=4,
                max_attempts=max_attempts,
            )
        )
    for period in _financial_periods(start, end):
        specs.extend(
            _paged_specs(
                "fina_audit",
                "fina_audit_vip",
                {"period": period},
                group=f"fina_audit:{period}",
                page_size=1_000,
                max_pages=16,
                max_attempts=max_attempts,
            )
        )
        for business_type in ("P", "D", "I"):
            specs.extend(
                _paged_specs(
                    "fina_mainbz",
                    "fina_mainbz_vip",
                    {"period": period, "type": business_type},
                    group=f"fina_mainbz:{period}:{business_type}",
                    page_size=1_000,
                    max_pages=64,
                    max_attempts=max_attempts,
                )
            )
    return specs


def _cn_fund_specs(start: date, end: date, dates: list[str], max_attempts: int) -> list[FetchSpec]:
    """Plan the public-fund research surface without per-fund request loops."""
    # fund_basic defaults to exchange-traded funds only.  Plan both markets so
    # this independently runnable bundle also has the master rows required to
    # interpret NAV, share, dividend and portfolio records.
    specs: list[FetchSpec] = []
    for market in ("E", "O"):
        for status in ("L", "I", "D"):
            if market == "O" and status == "L":
                specs.extend(
                    _paged_specs(
                        "fund_basic",
                        "fund_basic",
                        {"market": market, "status": status},
                        group=f"fund_basic:{market}:{status}",
                        page_size=5_000,
                        max_pages=16,
                        max_attempts=max_attempts,
                    )
                )
            else:
                specs.append(
                    _spec(
                        "fund_basic",
                        "fund_basic",
                        {"market": market, "status": status},
                        scope={"market": market, "status": status, "row_limit": 15_000},
                        allow_empty=True,
                        max_attempts=max_attempts,
                    )
                )
    for status in ("L", "D", "P"):
        specs.extend(
            _paged_specs(
                "etf_basic",
                "etf_basic",
                {"list_status": status},
                group=f"etf_basic:{status}",
                page_size=5_000,
                max_pages=4,
                max_attempts=max_attempts,
            )
        )
    specs.extend(
        _paged_specs(
            "etf_index",
            "etf_index",
            {},
            group="etf_index:all",
            page_size=5_000,
            max_pages=4,
            max_attempts=max_attempts,
        )
    )
    for dataset, page_size, max_pages in (
        ("fund_company", 1_000, 4),
        ("fund_manager", 2_000, 64),
    ):
        specs.extend(
            _paged_specs(
                dataset,
                dataset,
                {},
                group=f"{dataset}:all",
                page_size=page_size,
                max_pages=max_pages,
                max_attempts=max_attempts,
            )
        )
    fund_dates = dates or _weekdays(start, end)
    specs.extend(
        _daily_paged_specs(
            "fund_nav",
            "fund_nav",
            fund_dates,
            date_param="nav_date",
            date_field="nav_date",
            page_size=2_000,
            max_pages=64,
            max_attempts=max_attempts,
        )
    )
    for window_start, window_end in _month_ranges(start, end):
        window_dates = [
            value
            for value in fund_dates
            if compact_date(window_start) <= value <= compact_date(window_end)
        ]
        params = {
            "start_date": compact_date(window_start),
            "end_date": compact_date(window_end),
        }
        specs.extend(
            _paged_specs(
                "fund_share",
                "fund_share",
                params,
                group=f"fund_share:{params['start_date']}:{params['end_date']}",
                page_size=2_000,
                max_pages=64,
                max_attempts=max_attempts,
                expected_date_field="trade_date",
                partition=partition_metadata(
                    "date",
                    window_start,
                    window_end,
                    values=window_dates,
                ),
            )
        )
    announcement_dates = _calendar_dates(start, end)
    for dataset, page_size, max_pages in (
        ("fund_div", 1_000, 32),
        ("fund_portfolio", 2_000, 1_024),
    ):
        specs.extend(
            _daily_paged_specs(
                dataset,
                dataset,
                announcement_dates,
                date_param="ann_date",
                date_field="ann_date",
                page_size=page_size,
                max_pages=max_pages,
                max_attempts=max_attempts,
            )
        )
    return specs


def _cn_macro_specs(start: date, end: date, max_attempts: int) -> list[FetchSpec]:
    start_month = start.strftime("%Y%m")
    end_month = end.strftime("%Y%m")
    monthly = {"start_m": start_month, "end_m": end_month}
    specs = [
        _spec("cn_gdp", "cn_gdp", {}, allow_empty=False, max_attempts=max_attempts),
        *[
            _spec(name, name, dict(monthly), max_attempts=max_attempts)
            for name in ("cn_cpi", "cn_ppi", "cn_pmi", "cn_m", "sf_month")
        ],
    ]
    for name in ("shibor", "shibor_lpr"):
        history_range = clip_history_range(name, start, end)
        if history_range is None:
            continue
        daily = {
            "start_date": compact_date(history_range[0]),
            "end_date": compact_date(history_range[1]),
        }
        specs.append(_spec(name, name, daily, max_attempts=max_attempts))
    for month_start, _month_end in _month_ranges(start, end):
        month = month_start.strftime("%Y%m")
        specs.append(
            _spec(
                "cn_schedule",
                "cn_schedule",
                {"m": month},
                scope={"m": month, "row_limit": 3_000},
                max_attempts=max_attempts,
            )
        )
    return specs


_MAJOR_NEWS_SOURCES = (
    "新华网",
    "凤凰财经",
    "同花顺",
    "新浪财经",
    "华尔街见闻",
    "中证网",
    "财新网",
    "第一财经",
    "财联社",
)


def _cn_institutional_specs(
    start: date, end: date, dates: list[str], max_attempts: int
) -> list[FetchSpec]:
    """Plan the remaining Tushare-provided institutional research surface.

    Every capped endpoint uses an explicit limit/offset termination proof.  The
    request grain follows the provider's largest safe cross-section: monthly
    for research reports and Shibor quotes, and full-market trading-day pages
    for ETF baskets and industry indices.  Long-form news is source/day paged
    because its documented 400-row ceiling is too small for multi-day windows.
    """

    specs: list[FetchSpec] = []
    for window_start, window_end in _month_ranges(start, end):
        compact_start = compact_date(window_start)
        compact_end = compact_date(window_end)
        specs.extend(
            _paged_specs(
                "report_rc",
                "report_rc",
                {"start_date": compact_start, "end_date": compact_end},
                group=f"report_rc:{compact_start}:{compact_end}",
                page_size=3_000,
                max_pages=64,
                max_attempts=max_attempts,
            )
        )
        specs.append(
            _spec(
                "shibor_quote",
                "shibor_quote",
                {"start_date": compact_start, "end_date": compact_end},
                scope={
                    "start_date": compact_start,
                    "end_date": compact_end,
                    "row_limit": 4_000,
                },
                max_attempts=max_attempts,
            )
        )
    news_range = clip_history_range("major_news", start, end)
    for window_start, window_end in (
        _month_ranges(*news_range) if news_range is not None else ()
    ):
        compact_start = compact_date(window_start)
        compact_end = compact_date(window_end)
        start_text = f"{window_start.isoformat()} 00:00:00"
        end_text = f"{window_end.isoformat()} 23:59:59"
        for source in _MAJOR_NEWS_SOURCES:
            specs.extend(
                _paged_specs(
                    "major_news",
                    "major_news",
                    {"src": source, "start_date": start_text, "end_date": end_text},
                    group=f"major_news:{source}:{compact_start}:{compact_end}",
                    page_size=400,
                    max_pages=512,
                    max_attempts=max_attempts,
                    fields=("title", "content", "pub_time", "src"),
                )
            )

    market_dates = dates or _weekdays(start, end)
    for status in ("L", "D", "P"):
        specs.extend(
            _paged_specs(
                "etf_basic",
                "etf_basic",
                {"list_status": status},
                group=f"etf_basic:{status}",
                page_size=5_000,
                max_pages=4,
                max_attempts=max_attempts,
            )
        )
    specs.extend(
        _daily_paged_specs(
            "ci_daily",
            "ci_daily",
            market_dates,
            page_size=4_000,
            max_pages=4,
            max_attempts=max_attempts,
        )
    )

    return specs


def _cn_futures_specs(
    start: date, end: date, dates: list[str], max_attempts: int
) -> list[FetchSpec]:
    specs = [
        _spec(
            "fut_basic",
            "fut_basic",
            {"exchange": exchange},
            allow_empty=True,
            max_attempts=max_attempts,
        )
        for exchange in _CN_EXCHANGES
    ]
    for exchange in _CN_EXCHANGES:
        specs.append(
            _spec(
                "fut_trade_cal",
                "fut_trade_cal",
                {
                    "exchange": exchange,
                    "start_date": compact_date(start),
                    "end_date": compact_date(end),
                },
                allow_empty=False,
                max_attempts=max_attempts,
            )
        )
    specs.extend(
        _daily_paged_specs(
            "fut_mapping",
            "fut_mapping",
            dates,
            page_size=1_000,
            max_pages=8,
            max_attempts=max_attempts,
        )
    )
    for dataset, page_size, max_pages in (
        ("fut_daily", 1_000, 4),
        ("fut_holding", 1_000, 64),
        ("fut_wsr", 1_000, 4),
        ("fut_settle", 1_000, 4),
        ("ft_limit", 4_000, 4),
    ):
        specs.extend(
            _daily_paged_specs(
                dataset,
                dataset,
                dates,
                page_size=page_size,
                max_pages=max_pages,
                max_attempts=max_attempts,
            )
        )
    return specs


def _cn_options_bonds_specs(dates: list[str], max_attempts: int) -> list[FetchSpec]:
    specs = _paged_specs(
        "opt_basic",
        "opt_basic",
        {},
        group="opt_basic:all",
        page_size=2_000,
        max_pages=1_024,
        max_attempts=max_attempts,
    )
    specs.extend(
        _paged_specs(
            "cb_basic",
            "cb_basic",
            {},
            group="cb_basic:all",
            page_size=1_000,
            max_pages=6,
            max_attempts=max_attempts,
        )
    )
    for dataset, api_name, page_size, max_pages in (
        ("cb_issue", "cb_issue", 1_000, 8),
        # Tushare's public interface is named cb_call; keep cb_redeem as the
        # stable logical dataset name used by the platform catalog.
        ("cb_redeem", "cb_call", 1_000, 8),
    ):
        specs.extend(
            _paged_specs(
                dataset,
                api_name,
                {},
                group=f"{dataset}:all",
                page_size=page_size,
                max_pages=max_pages,
                max_attempts=max_attempts,
            )
        )
    specs.extend(
        _daily_paged_specs(
            "opt_daily",
            "opt_daily",
            dates,
            page_size=2_000,
            max_pages=64,
            max_attempts=max_attempts,
        )
    )
    for dataset in ("cb_daily", "repo_daily"):
        specs.extend(
            _daily_paged_specs(
                dataset,
                dataset,
                dates,
                page_size=1_000,
                max_pages=4,
                max_attempts=max_attempts,
            )
        )
    specs.extend(
        _daily_paged_specs(
            "yc_cb",
            "yc_cb",
            dates,
            page_size=1_000,
            max_pages=4,
            max_attempts=max_attempts,
        )
    )
    return specs


def bond_reference_specs(
    symbols: Iterable[str],
    *,
    start: date,
    end: date,
    max_attempts: int,
) -> list[FetchSpec]:
    """Plan code-required convertible-bond interfaces after cb_basic is present."""
    codes = sorted({str(value).strip() for value in symbols if str(value).strip()})
    if not codes:
        raise ValueError("at least one convertible-bond symbol is required")
    specs: list[FetchSpec] = []
    # These endpoints explicitly require ts_code.  Batching preserves the
    # provider contract without falling back to one request per bond.
    for dataset, api_name, batch_size, row_limit, date_bounded in (
        ("cb_rate", "cb_rate", 100, 2_000, False),
        ("cb_price_chg", "cb_price_chg", 100, 2_000, False),
        ("cb_rating", "cb_rating", 100, 3_000, False),
        ("top10_cb_holders", "top10_cb_holders", 100, 3_000, True),
        # Conversion results may contain daily observations, so use smaller
        # batches for a multi-year window.
        ("cb_share", "cb_share", 10, 2_000, True),
    ):
        for batch in _batched(codes, batch_size):
            params = {"ts_code": ",".join(batch)}
            if date_bounded:
                params.update({"start_date": compact_date(start), "end_date": compact_date(end)})
            specs.append(
                _spec(
                    dataset,
                    api_name,
                    params,
                    scope={**params, "row_limit": row_limit},
                    max_attempts=max_attempts,
                )
            )
    return apply_reference_refresh(specs, as_of=end)


def _hk_market_specs(start: date, end: date, max_attempts: int) -> list[FetchSpec]:
    specs: list[FetchSpec] = []
    for status in ("L", "D", "P"):
        specs.extend(
            _paged_specs(
                "hk_basic",
                "hk_basic",
                {"list_status": status},
                group=f"hk_basic:{status}",
                page_size=1_000,
                max_pages=8,
                max_attempts=max_attempts,
            )
        )
    specs.append(
        _spec(
            "hk_tradecal",
            "hk_tradecal",
            {"start_date": compact_date(start), "end_date": compact_date(end)},
            allow_empty=False,
            max_attempts=max_attempts,
        )
    )
    return specs


def _us_market_specs(start: date, end: date, max_attempts: int) -> list[FetchSpec]:
    specs = _paged_specs(
        "us_basic",
        "us_basic",
        {},
        group="us_basic:all",
        page_size=1_000,
        max_pages=30,
        max_attempts=max_attempts,
    )
    specs.append(
        _spec(
            "us_tradecal",
            "us_tradecal",
            {"start_date": compact_date(start), "end_date": compact_date(end)},
            allow_empty=False,
            max_attempts=max_attempts,
        )
    )
    return specs


def market_daily_specs(
    market: str,
    open_dates: Iterable[str],
    *,
    max_attempts: int,
) -> list[FetchSpec]:
    """Plan HK/US daily pages only after the official calendar is available."""

    if market not in {"hk", "us"}:
        raise ValueError("market must be hk or us")
    dates = sorted(set(open_dates))
    page_size, max_pages = ((1_000, 8) if market == "hk" else (2_000, 16))
    specs: list[FetchSpec] = []
    for dataset in (f"{market}_daily", f"{market}_daily_adj"):
        specs.extend(
            _daily_paged_specs(
                dataset,
                dataset,
                dates,
                page_size=page_size,
                max_pages=max_pages,
                max_attempts=max_attempts,
            )
        )
    return specs


def market_financial_specs(
    market: str,
    symbols: Iterable[str],
    *,
    start: date,
    end: date,
    max_attempts: int,
) -> list[FetchSpec]:
    """Plan the largest financial-history request shape supported by each market."""
    if market not in {"hk", "us"}:
        raise ValueError("market must be hk or us")
    if market == "us":
        specs: list[FetchSpec] = []
        for period in _financial_periods(start, end):
            for dataset, api_name in (
                ("us_income", "us_income_vip"),
                ("us_balancesheet", "us_balancesheet_vip"),
                ("us_cashflow", "us_cashflow_vip"),
            ):
                specs.extend(
                    _paged_specs(
                        dataset,
                        api_name,
                        {"period": period},
                        group=f"{dataset}:{period}",
                        page_size=1_000,
                        max_pages=512,
                        max_attempts=max_attempts,
                        expected_date_field="end_date",
                        expected_date=period,
                    )
                )
        for symbol in sorted({str(value).strip() for value in symbols if str(value).strip()}):
            params = {
                "ts_code": symbol,
                "start_date": compact_date(start),
                "end_date": compact_date(end),
            }
            specs.append(
                _spec(
                    "us_fina_indicator",
                    "us_fina_indicator",
                    params,
                    scope={**params, "row_limit": 200},
                    max_attempts=max_attempts,
                )
            )
        return specs
    endpoints = (
        "hk_income",
        "hk_balancesheet",
        "hk_cashflow",
        "hk_fina_indicator",
    )
    specs: list[FetchSpec] = []
    for symbol in sorted({str(value).strip() for value in symbols if str(value).strip()}):
        for dataset in endpoints:
            params = {
                "ts_code": symbol,
                "start_date": compact_date(start),
                "end_date": compact_date(end),
            }
            specs.append(
                _spec(
                    dataset,
                    dataset,
                    params,
                    scope={**params, "row_limit": 10_000},
                    max_attempts=max_attempts,
                )
            )
    return specs


def _global_market_specs(start: date, end: date, max_attempts: int) -> list[FetchSpec]:
    specs: list[FetchSpec] = []
    specs.extend(
        _paged_specs(
            "fx_obasic",
            "fx_obasic",
            {},
            group="fx_obasic:all",
            page_size=1_000,
            max_pages=8,
            max_attempts=max_attempts,
        )
    )
    dates = _weekdays(start, end)
    for dataset, page_size, max_pages in (
        ("index_global", 2_000, 8),
        ("fx_daily", 1_000, 8),
    ):
        specs.extend(
            _daily_paged_specs(
                dataset,
                dataset,
                dates,
                page_size=page_size,
                max_pages=max_pages,
                max_attempts=max_attempts,
            )
        )
    for window_start, window_end in _year_ranges(start, end):
        params = {
            "start_date": compact_date(window_start),
            "end_date": compact_date(window_end),
        }
        specs.append(
            _spec(
                "us_tycr",
                "us_tycr",
                params,
                scope={**params, "row_limit": 2_000},
                max_attempts=max_attempts,
            )
        )
    return specs


def _month_ranges(start: date, end: date) -> list[tuple[date, date]]:
    ranges: list[tuple[date, date]] = []
    current = start
    while current <= end:
        last = date(current.year, current.month, monthrange(current.year, current.month)[1])
        window_end = min(last, end)
        ranges.append((current, window_end))
        current = window_end + timedelta(days=1)
    return ranges


def _year_ranges(start: date, end: date) -> list[tuple[date, date]]:
    ranges: list[tuple[date, date]] = []
    current = start
    while current <= end:
        window_end = min(date(current.year, 12, 31), end)
        ranges.append((current, window_end))
        current = window_end + timedelta(days=1)
    return ranges


def _weekdays(start: date, end: date) -> list[str]:
    result: list[str] = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            result.append(compact_date(current))
        current += timedelta(days=1)
    return result


def _calendar_dates(start: date, end: date) -> list[str]:
    result: list[str] = []
    current = start
    while current <= end:
        result.append(compact_date(current))
        current += timedelta(days=1)
    return result


def _financial_periods(start: date, end: date) -> list[str]:
    quarter_ends = [
        date(year, month, day)
        for year in range(start.year - 2, end.year + 1)
        for month, day in ((3, 31), (6, 30), (9, 30), (12, 31))
    ]
    previous_four = sorted(period for period in quarter_ends if period < start)[-4:]
    requested = [period for period in quarter_ends if start <= period <= end]
    return [compact_date(period) for period in sorted({*previous_four, *requested})]


def _batched(values: list[str], size: int) -> Iterable[list[str]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]
