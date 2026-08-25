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
