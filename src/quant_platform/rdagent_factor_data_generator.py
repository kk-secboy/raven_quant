"""Generate RD-Agent factor implementation samples from the governed Qlib data."""

from __future__ import annotations

import qlib

qlib.init(provider_uri="~/.qlib/qlib_data/cn_data")

from qlib.data import D  # noqa: E402

FIELDS = ["$open", "$close", "$high", "$low", "$volume", "$factor"]
instruments = D.instruments()

full = (
    D.features(instruments, FIELDS, freq="day")
    .swaplevel()
    .sort_index()
    .loc["2008-12-29":]
    .sort_index()
)
if full.empty:
    raise RuntimeError("governed Qlib dataset produced no factor implementation rows")
full.to_hdf("./daily_pv_all.h5", key="data")

debug = (
    D.features(
        instruments,
        FIELDS,
        start_time="2018-01-01",
        end_time="2019-12-31",
        freq="day",
    )
    .swaplevel()
    .sort_index()
)
if debug.empty:
    raise RuntimeError("governed Qlib dataset produced no debug factor rows")
available = debug.index.get_level_values("instrument").unique()[:100]
debug = debug[debug.index.get_level_values("instrument").isin(available)]
debug.to_hdf("./daily_pv_debug.h5", key="data")
