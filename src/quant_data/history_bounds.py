from __future__ import annotations

from datetime import date

# Tushare documents both ``news`` and ``major_news`` with history beginning
# around 2018-11-20.  Keep source-history bounds separate from point-in-time
# availability rules: this is about whether a provider can return a historical
# row at all, not when a returned row became knowable to a researcher.
TUSHARE_NEWS_HISTORY_START = date(2018, 11, 20)


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
    lower_bound = {
        "news": TUSHARE_NEWS_HISTORY_START,
        "major_news": TUSHARE_NEWS_HISTORY_START,
    }.get(dataset)
    clipped_start = max(start, lower_bound) if lower_bound is not None else start
    if end < clipped_start:
        return None
    return clipped_start, end
