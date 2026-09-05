"""Retain only the exact Qlib panels consumed by the standalone baseline."""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandler, DataHandlerLP
from qlib.data.dataset.utils import fetch_df_by_col


def _retain_rows(frame: pd.DataFrame, periods) -> pd.DataFrame:
    dates = frame.index.get_level_values("datetime")
    keep = np.zeros(len(frame), dtype=bool)
    for start, end in periods:
        keep |= (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    if keep.all():
        return frame
    # A continuous .loc slice can keep the entire original block alive. A deep
    # copy can consolidate feature/label blocks into several full-size buffers.
    # Row take copies just the retained rows of each existing block.
    return frame.take(np.flatnonzero(keep))


class BaselineDataset(DatasetH):
    """Preserve train/valid learning data and test inference data, exactly.

    Construct only after all original Alpha158 processing has completed. This
    dataset intentionally supports only the standalone baseline's consumers;
    it cannot become a general handler that silently omits another data range.
    """

    def __init__(self, *, handler: DataHandlerLP, segments: dict) -> None:
        if set(segments) != {"train", "valid", "test"}:
            raise ValueError("baseline requires exactly train, valid and test segments")
        periods = [tuple(pd.Timestamp(value) for value in segments[key])
                   for key in ("train", "valid", "test")]
        if any(len(period) != 2 or period[0] > period[1] for period in periods):
            raise ValueError("baseline segment bounds are invalid")
        if not periods[0][1] < periods[1][0] or not periods[1][1] < periods[2][0]:
            raise ValueError("baseline segments must be ordered and disjoint")
        if getattr(handler, "_data", None) is not handler._infer:
            raise ValueError("baseline requires the unmodified Alpha158 inference panel")
        handler._learn = _retain_rows(handler._learn, periods[:2])
        handler._infer = _retain_rows(handler._infer, periods[2:])
        del handler._data
        handler.drop_raw = True
        gc.collect()
        super().__init__(handler=handler, segments=segments)

    def prepare(self, segments, col_set=DataHandler.CS_ALL,
                data_key=DataHandlerLP.DK_I, **kwargs):
        names = [segments] if isinstance(segments, str) else segments
        learning = (
            isinstance(names, (list, tuple)) and bool(names)
            and all(name in ("train", "valid") for name in names)
            and data_key == DataHandlerLP.DK_L and col_set == ["feature", "label"]
        )
        inference = (
            names == ["test"] and data_key == DataHandlerLP.DK_I
            and isinstance(col_set, str) and col_set in ("feature", "label")
        )
        if kwargs or not (learning or inference):
            raise ValueError("unsupported standalone baseline dataset consumer")
        return super().prepare(segments, col_set=col_set, data_key=data_key)

    def _prepare_seg(self, slc, **kwargs):
        col_set = kwargs.pop("col_set")
        # Qlib's default handler selects columns across all dates before slicing
        # the requested segment. CS_RAW bypasses that selection; the unchanged
        # upstream column selector then sees only this segment's rows.
        frame = super()._prepare_seg(slc, col_set=DataHandler.CS_RAW, **kwargs)
        return fetch_df_by_col(frame, col_set)
