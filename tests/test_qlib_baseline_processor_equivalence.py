"""Run with pytest or unittest inside the pinned Qlib runtime; never use a DB."""

from __future__ import annotations

import importlib.util
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

try:
    import pytest
except ImportError:  # The production Qlib runtime need not install pytest.
    pass
else:
    pytestmark = pytest.mark.no_database

QLIB_AVAILABLE = importlib.util.find_spec("qlib") is not None
if QLIB_AVAILABLE:
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset.processor import Processor

    from quant_platform.qlib_baseline_processors import BaselineLabelProcessor


@unittest.skipUnless(QLIB_AVAILABLE, "requires the installed pinned Qlib runtime")
class BaselineProcessorEquivalenceTests(unittest.TestCase):
    @staticmethod
    def frame(dtype, missing="some", reverse=False):
        index = pd.MultiIndex.from_product(
            [pd.date_range("2026-01-05", periods=5), ["SH600001", "SH600002", "SH600003"]],
            names=["datetime", "instrument"],
        )
        columns = pd.MultiIndex.from_tuples(
            [("feature", f"F{i:03d}") for i in range(158)] + [("label", "LABEL0")]
        )
        values = np.arange(len(index) * len(columns), dtype=dtype).reshape(len(index), -1)
        values /= dtype(37)
        values[0, 0] = np.nan
        values[1, 1] = np.inf
        values[2, 2] = -np.inf
        values[3, 3] = -0.0
        # Ordinary, constant, infinite and singleton label cross sections.
        values[:, -1] = np.asarray(
            [0.1, 0.2, 0.5, 1, 1, 1, np.inf, 0, -np.inf, 2, 3, 4, 0.2, 0.3, 0.8],
            dtype=dtype,
        )
        if missing == "some":
            values[[1, 9, 10, 12, 13, 14], -1] = np.nan
        elif missing == "all":
            values[:, -1] = np.nan
        frame = pd.DataFrame(values, index=index, columns=columns)
        # Qlib concatenates feature and label loader outputs as separate blocks.
        result = pd.concat({"feature": frame["feature"], "label": frame["label"]}, axis=1)
        return result.iloc[::-1] if reverse else result

    @staticmethod
    def handler(frame, bounded):
        kwargs = {}
        if bounded:
            kwargs["learn_processors"] = [
                {"class": "BaselineLabelProcessor",
                 "module_path": "quant_platform.qlib_baseline_processors"}
            ]
        handler = Alpha158(init_data=False, **kwargs)
        handler._data = frame
        return handler

    def assert_exact_frame(self, expected, actual):
        pd.testing.assert_frame_equal(expected, actual, check_exact=True, check_like=False)
        for column in expected.columns:
            left = expected[column].to_numpy()
            right = actual[column].to_numpy()
            np.testing.assert_array_equal(np.isnan(left), np.isnan(right))
            np.testing.assert_array_equal(np.isposinf(left), np.isposinf(right))
            np.testing.assert_array_equal(np.isneginf(left), np.isneginf(right))
            # Also preserve signed zero and the exact floating-point bit pattern.
            self.assertEqual(left.tobytes(), right.tobytes())

    def test_matches_real_upstream_chain_and_never_mutates_input(self):
        for dtype in (np.float32, np.float64):
            for missing in ("some", "none", "all"):
                for reverse in (False, True):
                    for copy_on_write in (False, True):
                        with self.subTest(dtype=dtype, missing=missing, reverse=reverse,
                                          copy_on_write=copy_on_write):
                            with pd.option_context("mode.copy_on_write", copy_on_write):
                                source = self.frame(dtype, missing, reverse)
                                original = source.copy(deep=True)
                                reference = self.handler(source, bounded=False)
                                reference.process_data(with_fit=True)
                                bounded = self.handler(source, bounded=True)
                                bounded.process_data(with_fit=True)
                                self.assert_exact_frame(original, source)
                                self.assert_exact_frame(reference._infer, bounded._infer)
                                self.assert_exact_frame(reference._learn, bounded._learn)
                                self.assertIs(bounded._infer, source)
                                self.assertIsNot(bounded._learn, source)
                                if not bounded._learn.empty:
                                    bounded._learn.iloc[0, 0] = 12345
                                    self.assert_exact_frame(original, source)

    def test_avoids_handler_full_input_copy(self):
        copy = pd.DataFrame.copy
        counts = []
        for bounded in (False, True):
            source = self.frame(np.float32)
            handler = self.handler(source, bounded)
            copies = []

            def record_copy(frame, *args, _source=source, _copies=copies, **kwargs):
                if frame is _source:
                    _copies.append(kwargs.get("deep", True))
                return copy(frame, *args, **kwargs)

            with patch.object(pd.DataFrame, "copy", record_copy):
                handler.process_data(with_fit=True)
            counts.append(copies)
        self.assertEqual(counts, [[True], []])

    def test_pinned_default_chain_and_learning_only_contract(self):
        reference = Alpha158(init_data=False)
        self.assertEqual(
            [(type(p).__name__, p.fields_group) for p in reference.learn_processors],
            [("DropnaLabel", "label"), ("CSZScoreNorm", "label")],
        )
        self.assertEqual(reference.shared_processors, [])
        self.assertEqual(reference.infer_processors, [])
        self.assertFalse(reference.drop_raw)
        self.assertTrue(all(type(p).fit is Processor.fit for p in reference.learn_processors))
        processor = BaselineLabelProcessor()
        self.assertTrue(processor.readonly())
        self.assertFalse(processor.is_for_infer())


if __name__ == "__main__":
    unittest.main()
