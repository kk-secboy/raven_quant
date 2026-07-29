from __future__ import annotations

from datetime import date

# Keep source-history bounds separate from point-in-time availability rules:
# this is about whether a provider can return a historical row at all, not when
# a returned row became knowable to a researcher.  Every bound below is taken
# from the provider's interface/permission documentation; unknown datasets stay
# unbounded instead of guessing an inception date.
TUSHARE_NEWS_HISTORY_START = date(2018, 11, 20)
TUSHARE_HISTORY_STARTS: dict[str, date] = {
    "news": TUSHARE_NEWS_HISTORY_START,
    "major_news": TUSHARE_NEWS_HISTORY_START,
    "moneyflow": date(2010, 1, 1),
    "margin_detail": date(2010, 1, 1),
    "repurchase": date(2011, 1, 1),
    "pledge_stat": date(2014, 1, 1),
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
