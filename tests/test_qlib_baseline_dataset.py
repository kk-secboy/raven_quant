"""Exact baseline consumer and training comparisons in the pinned Qlib runtime."""

from __future__ import annotations

import gc
import importlib.util
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

try:
    import pytest
except ImportError:
    pass
else:
    pytestmark = pytest.mark.no_database

QLIB_AVAILABLE = importlib.util.find_spec("qlib") is not None
if QLIB_AVAILABLE:
    from qlib.contrib.data.handler import Alpha158
    from qlib.contrib.model import gbdt
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP

    from quant_platform.qlib_baseline_dataset import BaselineDataset


def synthetic_handler(*, days=120, instruments=12, dtype=np.float32, nonfinite=False):
    dates = pd.date_range("2020-01-01", periods=days)
    index = pd.MultiIndex.from_product(
        [dates, [f"stock{i:04d}" for i in range(instruments)]],
        names=["datetime", "instrument"],
    )
    rng = np.random.default_rng(20260905)
    values = rng.standard_normal((len(index), 158)).astype(dtype)
    labels = (values[:, 0] * dtype(0.2) + values[:, 1] * dtype(0.1)).copy()
    labels[::19] = np.nan
    values[::23, 7] = np.nan
    if nonfinite:
        values[1, 2], values[2, 3], values[3, 4] = np.inf, -np.inf, -0.0
        labels[4], labels[5] = np.inf, -np.inf
    frame = pd.concat({
        "feature": pd.DataFrame(values, index=index, columns=[f"F{i:03d}" for i in range(158)]),
        "label": pd.DataFrame(labels, index=index, columns=["LABEL0"]),
    }, axis=1)
    handler = Alpha158(init_data=False, learn_processors=[{
        "class": "BaselineLabelProcessor",
        "module_path": "quant_platform.qlib_baseline_processors",
    }])
    handler._data = frame
    handler.process_data(with_fit=True)
    train_end, valid_end = int(days * 0.6), int(days * 0.8)
    segments = {"train": (dates[0], dates[train_end - 1]),
                "valid": (dates[train_end], dates[valid_end - 1]),
                "test": (dates[valid_end], dates[-1])}
    return handler, segments


@unittest.skipUnless(QLIB_AVAILABLE, "requires the installed pinned Qlib runtime")
class BaselineDatasetTests(unittest.TestCase):
    def assert_exact_frame(self, expected, actual):
        pd.testing.assert_frame_equal(expected, actual, check_exact=True)
        for column in expected.columns:
            left, right = expected[column].to_numpy(), actual[column].to_numpy()
            np.testing.assert_array_equal(np.isnan(left), np.isnan(right))
            np.testing.assert_array_equal(np.isposinf(left), np.isposinf(right))
            np.testing.assert_array_equal(np.isneginf(left), np.isneginf(right))
            self.assertEqual(left.tobytes(), right.tobytes())

    def test_all_consumers_exact_and_retained_blocks_detached(self):
        calls = [
            ("train", ["feature", "label"], DataHandlerLP.DK_L),
            ("valid", ["feature", "label"], DataHandlerLP.DK_L),
            ("test", "feature", DataHandlerLP.DK_I),
            ("test", "label", DataHandlerLP.DK_I),
        ]
        for dtype in (np.float32, np.float64):
            for copy_on_write in (False, True):
                with self.subTest(dtype=dtype, copy_on_write=copy_on_write):
                    with pd.option_context("mode.copy_on_write", copy_on_write):
                        handler, segments = synthetic_handler(dtype=dtype, nonfinite=True)
                        original = DatasetH(handler=handler, segments=segments)
                        expected = [original.prepare(seg, col_set=cols, data_key=key)
                                    for seg, cols, key in calls]
                        original_infer, original_learn = handler._infer, handler._learn
                        bounded = BaselineDataset(handler=handler, segments=segments)
                        for before, (seg, cols, key) in zip(expected, calls, strict=True):
                            self.assert_exact_frame(
                                before, bounded.prepare(seg, col_set=cols, data_key=key)
                            )
                        self.assert_exact_frame(expected[-1], bounded.prepare("test", "label"))
                        self.assertFalse(hasattr(handler, "_data"))
                        self.assertTrue(handler.drop_raw)
                        for before, after in ((original_infer, handler._infer),
                                              (original_learn, handler._learn)):
                            self.assertLess(len(after), len(before))
                            for original_block in before._mgr.blocks:
                                for retained_block in after._mgr.blocks:
                                    self.assertFalse(np.shares_memory(
                                        original_block.values, retained_block.values
                                    ))

    def test_rejects_unretained_consumers_and_overlapping_segments(self):
        handler, segments = synthetic_handler()
        bounded = BaselineDataset(handler=handler, segments=segments)
        for args, kwargs in [
            (("train",), {"col_set": "feature"}),
            (("test",), {"col_set": ["feature", "label"], "data_key": "learn"}),
            (("test",), {"col_set": "label", "squeeze": True}),
        ]:
            with self.assertRaisesRegex(ValueError, "unsupported standalone"):
                bounded.prepare(*args, **kwargs)
        handler, segments = synthetic_handler()
        segments["valid"] = segments["train"]
        with self.assertRaisesRegex(ValueError, "ordered and disjoint"):
            BaselineDataset(handler=handler, segments=segments)

    def test_old_panels_and_finished_inference_features_are_released(self):
        handler, segments = synthetic_handler()
        old_panels = [weakref.ref(handler._data), weakref.ref(handler._learn)]
        old_blocks = [weakref.ref(block.values)
                      for frame in (handler._data, handler._learn)
                      for block in frame._mgr.blocks]
        with patch.object(handler, "setup_data", side_effect=AssertionError("reloaded handler")):
            dataset = BaselineDataset(handler=handler, segments=segments)
        gc.collect()
        self.assertTrue(all(reference() is None for reference in old_panels + old_blocks))
        retained_blocks = [weakref.ref(block.values)
                           for frame in (handler._infer, handler._learn)
                           for block in frame._mgr.blocks]
        labels = dataset.prepare("test", col_set="label").copy(deep=True)
        expected = labels.copy(deep=True)
        del dataset, handler
        gc.collect()
        self.assertTrue(all(reference() is None for reference in retained_blocks))
        self.assert_exact_frame(expected, labels)

    def test_fixed_seed_upstream_lightgbm_training_predictions_and_ic_exact(self):
        results = []
        for bounded in (False, True):
            handler, segments = synthetic_handler()
            dataset_type = BaselineDataset if bounded else DatasetH
            dataset = dataset_type(handler=handler, segments=segments)
            model = gbdt.LGBModel(
                loss="mse", learning_rate=0.05, max_depth=6, num_leaves=63,
                colsample_bytree=0.8, subsample=0.8, lambda_l1=1.0, lambda_l2=1.0,
                num_threads=1, num_boost_round=20, early_stopping_rounds=5, seed=20260905,
            )
            metrics = {}
            # Only the recorder is replaced: no database or local MLflow run.
            with patch.object(gbdt, "R", SimpleNamespace(log_metrics=lambda **_kwargs: None)):
                model.fit(dataset, evals_result=metrics, verbose_eval=False)
            predictions = model.predict(dataset, segment="test").rename("score")
            labels = dataset.prepare("test", col_set="label").iloc[:, 0].rename("label")
            aligned = pd.concat([predictions, labels], axis=1).dropna()
            ic = aligned.groupby(level="datetime").apply(
                lambda frame: frame["score"].corr(frame["label"]), include_groups=False
            )
            rank_ic = aligned.groupby(level="datetime").apply(
                lambda frame: frame["score"].corr(frame["label"], method="spearman"),
                include_groups=False,
            )
            results.append((metrics, model.model.best_iteration, predictions, aligned, ic, rank_ic))
        self.assertEqual(results[0][:2], results[1][:2])
        self.assert_exact_frame(results[0][2].to_frame(), results[1][2].to_frame())
        self.assert_exact_frame(results[0][3], results[1][3])
        for index in (4, 5):
            pd.testing.assert_series_equal(results[0][index], results[1][index], check_exact=True)


if __name__ == "__main__":
    unittest.main()
