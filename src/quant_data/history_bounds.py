from __future__ import annotations

from datetime import date

# Keep source-history bounds separate from point-in-time availability rules:
# this is about whether a provider can return a historical row at all, not when
# a returned row became knowable to a researcher.  Every bound below is taken
# from the provider's interface/permission documentation; unknown datasets stay
# unbounded instead of guessing an inception date.
TUSHARE_NEWS_HISTORY_START = date(2018, 11, 20)

# The configured primary Tushare-compatible gateway is the production source
# from 2016 onward for the three market series that BaoStock backfills for
# 2008-2015.  Keeping this boundary in the planner prevents both providers from
# being requested for the same natural primary key.  Immutable units produced
# before this contract was encoded remain on disk for audit, but successor
# release selection retires those out-of-contract primary requests.
PRIMARY_MARKET_HISTORY_START = date(2016, 1, 1)
PRIMARY_MARKET_HISTORY_DATASETS = frozenset(
    {"daily", "daily_basic", "adj_factor"}
)

# Tushare only exposes a complete, internally consistent Beijing Stock
# Exchange cross-section from 2023 onward.  Earlier quote rows can use later
# code mappings and do not have matching daily_basic / price-limit history, so
# they cannot form a governed investable or model-training universe.
BSE_GOVERNED_HISTORY_START = date(2023, 1, 1)
GOVERNED_DAILY_STOCK_SCOPE_VERSION = "cn-mainland-a-share-daily-scope-v1"

# Freeze the exchange/code families admitted by the governed A-share product.
# ``stock_basic`` is necessary lifecycle evidence, but it is a current-only
# provider surface and can omit absorbed or historically delisted codes.  Code
# shape therefore remains an independent fail-closed type check when daily and
# daily_basic jointly prove a historical security that the current master lost.
_MAINLAND_A_SHARE_PREFIXES: dict[str, tuple[str, ...]] = {
    "SH": ("600", "601", "603", "605", "688", "689"),
    "SZ": ("000", "001", "002", "003", "300", "301"),
    # Beijing common-share histories can retain either their historical code
    # family or the later 920-series mapping.  Their usable date boundary is
    # enforced separately by ``BSE_GOVERNED_HISTORY_START``.
    "BJ": ("43", "83", "87", "88", "92"),
}


def is_mainland_b_share_code(ts_code: str) -> bool:
    """Return whether a Tushare security code is an out-of-scope B share."""

    normalized = str(ts_code).strip().upper()
    code, separator, exchange = normalized.partition(".")
    if not separator or len(code) != 6 or not code.isdigit():
        return False
    return (exchange == "SZ" and code.startswith("20")) or (
        exchange == "SH" and code.startswith("900")
    )


def is_governed_mainland_a_share_code(ts_code: str) -> bool:
    """Return whether a normalized code belongs to a governed A-share family."""

    normalized = str(ts_code).strip().upper()
    code, separator, exchange = normalized.partition(".")
    if (
        not separator
        or len(code) != 6
        or not code.isdigit()
        or is_mainland_b_share_code(normalized)
    ):
        return False
    prefixes = _MAINLAND_A_SHARE_PREFIXES.get(exchange)
    return bool(prefixes and code.startswith(prefixes))

TUSHARE_HISTORY_STARTS: dict[str, date] = {
    **{
        dataset: PRIMARY_MARKET_HISTORY_START
        for dataset in PRIMARY_MARKET_HISTORY_DATASETS
    },
    "news": TUSHARE_NEWS_HISTORY_START,
    "major_news": TUSHARE_NEWS_HISTORY_START,
    "report_rc": date(2010, 1, 1),
    # Official research_report documentation only offers history from this
    # date; planning an earlier request produces neither valid coverage nor a
    # meaningful empty-result proof.
    "research_report": date(2017, 1, 1),
    "moneyflow": date(2010, 1, 1),
    "margin_detail": date(2010, 1, 1),
    "repurchase": date(2011, 1, 1),
    "pledge_stat": date(2014, 1, 1),
    # The configured production gateway rejects report periods before 2016,
    # including prior-year periods whose actual disclosure happened later.
    "disclosure_date": date(2016, 1, 1),
    "stock_st": date(2016, 1, 1),
    "shibor": date(2006, 1, 1),
    "shibor_quote": date(2006, 1, 1),
    "shibor_lpr": date(2013, 1, 1),
}


def history_start_date(dataset: str) -> date | None:
    """Return the documented first possible source date, when one is known."""

    return TUSHARE_HISTORY_STARTS.get(dataset)


def clip_history_range(
    dataset: str,
    start: date,
    end: date,
) -> tuple[date, date] | None:
    """Clip a requested range to the provider's documented history.

    ``None`` means that the complete requested interval predates the source.
    Unknown datasets remain unchanged until their source boundary is verified.
    """

    if end < start:
        raise ValueError("end must not be before start")
    lower_bound = history_start_date(dataset)
    clipped_start = max(start, lower_bound) if lower_bound is not None else start
    if end < clipped_start:
        return None
    return clipped_start, end
