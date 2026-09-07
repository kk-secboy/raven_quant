"""Bound model-loading intermediates without changing Qlib's data transforms."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FEATURE_LOAD_BATCH_SIZE = 8


def _column_frame(columns: Mapping[tuple[str, str], np.ndarray], index: pd.Index):
    frame = pd.DataFrame(dict(columns), index=index, copy=False)
    frame.columns = pd.MultiIndex.from_tuples(list(columns))
    return frame


def build_model_handler_from_columns(
    columns: dict[tuple[str, str], np.ndarray],
    index: pd.MultiIndex,
    *,
    fit_start_time: str,
    fit_end_time: str,
    on_stage: Callable[[str], None] | None = None,
):
    """Consume raw columns and retain one shared normalized feature matrix.

    Qlib's feature transforms are column-independent. Its label transforms are
    applied separately, with the same complete date cross sections. No dates,
    instruments, expressions, precision, normalization or training rules change.
    """
    from qlib.data.dataset.handler import DataHandler, DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader
    from qlib.data.dataset.processor import CSZScoreNorm, DropnaLabel, Fillna, RobustZScoreNorm
    from qlib.data.dataset.storage import BaseHandlerStorage
    from qlib.data.dataset.utils import fetch_df_by_col, fetch_df_by_index

    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("model handler requires the original unique sorted Qlib index")
    features = {key: columns.pop(key) for key in list(columns) if key[0] == "feature"}
    if not features or not columns or any(key[0] != "label" for key in columns):
        raise ValueError("model handler requires feature and label groups")
    # Qlib RobustZScoreNorm fits df[feature_columns].values. Mixed feature
    # dtypes therefore use their common NumPy dtype even for an individual
    # float32 input column. Preserve that promotion across processing batches.
    feature_dtype = np.result_type(*(values.dtype for values in features.values()))
    keys = list(features)
    for offset in range(0, len(keys), FEATURE_LOAD_BATCH_SIZE):
        batch_keys = keys[offset : offset + FEATURE_LOAD_BATCH_SIZE]
        frame = _column_frame(
            {key: features[key].astype(feature_dtype, copy=False) for key in batch_keys}, index
        )
        normalizer = RobustZScoreNorm(
            fit_start_time=fit_start_time, fit_end_time=fit_end_time,
            fields_group="feature", clip_outlier=True,
        )
        normalizer.fit(frame)
        frame = Fillna(fields_group="feature")(normalizer(frame))
        for key in batch_keys:
            features[key] = frame[key].to_numpy(copy=False)
        del frame, normalizer
        if on_stage:
            on_stage(f"handler_normalized_features_{offset + len(batch_keys)}")
    raw_labels = _column_frame(columns, index)
    columns.clear()
    learn_labels = CSZScoreNorm(fields_group="label")(DropnaLabel()(raw_labels.copy()))
    if on_stage:
        on_stage("handler_labels_ready")

    class SharedFeatureStorage(BaseHandlerStorage):
        """Materialize only the requested segment; infer/learn share features."""

        def __init__(self, labels):
            self.labels = labels
            self.columns = pd.MultiIndex.from_tuples([*features, *list(labels.columns)])

        def fetch(self, selector=slice(None), level="datetime", col_set=DataHandler.CS_ALL,
                  fetch_orig=True):
            labels = fetch_df_by_index(self.labels, selector, level, fetch_orig=fetch_orig)
            selected_index = labels.index
            # Ask Qlib itself for the exact column-selection/order semantics.
            empty = pd.DataFrame(columns=self.columns)
            selected_columns = fetch_df_by_col(empty, col_set).columns
            if isinstance(selected_columns, pd.MultiIndex):
                selected_keys = list(selected_columns)
            elif col_set == DataHandler.CS_ALL:
                selected_keys = list(self.columns)
            else:
                selected_keys = [(str(col_set), name) for name in selected_columns]
            feature_keys = [key for key in selected_keys if key[0] == "feature"]
            if feature_keys:
                if selected_index.equals(index):
                    positions = slice(None)
                else:
                    positions = index.get_indexer(selected_index)
                    if (positions < 0).any():
                        raise ValueError("model segment escaped its frozen Qlib index")
                    if len(positions) and np.all(np.diff(positions) == 1):
                        positions = slice(int(positions[0]), int(positions[-1]) + 1)
                # A single block lets the original Qlib model consume .values
                # without consolidating hundreds of fragmented column blocks.
                matrix = np.empty((len(selected_index), len(feature_keys)),
                                  dtype=feature_dtype, order="F")
                for column, key in enumerate(feature_keys):
                    matrix[:, column] = features[key][positions]
                result = pd.DataFrame(matrix, index=selected_index,
                                      columns=pd.MultiIndex.from_tuples(feature_keys), copy=False)
            else:
                result = pd.DataFrame(index=selected_index)
            for position, key in enumerate(selected_keys):
                if key[0] == "label":
                    result.insert(position, key, labels[key].to_numpy(copy=False))
            result.columns = selected_columns
            return result

        def head(self, n=5):
            return self.fetch(self.labels.index[:n], level=None, col_set=DataHandler.CS_RAW)

    handler = DataHandlerLP(
        data_loader=StaticDataLoader(pd.DataFrame()), init_data=False, drop_raw=True,
    )
    handler._infer = SharedFeatureStorage(raw_labels)
    handler._learn = SharedFeatureStorage(learn_labels)
    return handler


def load_memory_bounded_model_handler(
    *,
    features: Mapping[str, str],
    label_expression: str,
    instruments: Any,
    start_time: str,
    end_time: str,
    fit_end_time: str,
    additional_factors_path: Path | None = None,
    on_stage: Callable[[str], None] | None = None,
):
    """Load feature batches with full expression context, then run Qlib processors."""
    from qlib.data.dataset.loader import QlibDataLoader, StaticDataLoader

    columns: dict[tuple[str, str], np.ndarray] = {}
    index = None
    names = list(features)
    for offset in range(0, len(names), FEATURE_LOAD_BATCH_SIZE):
        batch_names = names[offset : offset + FEATURE_LOAD_BATCH_SIZE]
        config = {"feature": [[features[name] for name in batch_names], batch_names]}
        if index is None:
            config["label"] = [[label_expression], ["LABEL0"]]
        frame = QlibDataLoader(config=config).load(instruments, start_time, end_time)
        if index is None:
            index = frame.index
        elif not frame.index.equals(index):
            raise ValueError("feature batching changed the governed Qlib row universe")
        for key in frame.columns:
            columns[key] = frame[key].to_numpy(copy=False)
        del frame
        if on_stage:
            on_stage(f"handler_loaded_features_{offset + len(batch_names)}")
    if index is None:
        raise ValueError("model handler has no governed features")
    if additional_factors_path is not None:
        # Qlib's group-config load_dataset path does not support parquet, while
        # its direct StaticDataLoader path does. Read the verified parquet once
        # and pass the DataFrame to retain the same group/left-join semantics.
        loader = StaticDataLoader(config={"feature": pd.read_parquet(additional_factors_path)})
        try:
            additional = loader.load(instruments, start_time, end_time)
        except KeyError:
            # Match Qlib NestedDataLoader's unsupported-market fallback.
            additional = loader.load(None, start_time, end_time)
        if not additional.index.is_unique:
            raise ValueError("additional factors contain duplicate governed rows")
        additional = additional.reindex(index)
        for key in additional.columns:
            columns[key] = additional[key].to_numpy(copy=False)
        # NestedDataLoader replaces duplicate columns and sorts its final columns.
        columns = dict(sorted(columns.items()))
        del additional, loader
    return build_model_handler_from_columns(
        columns, index, fit_start_time=start_time, fit_end_time=fit_end_time, on_stage=on_stage,
    )
