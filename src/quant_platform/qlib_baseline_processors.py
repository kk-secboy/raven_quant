"""Memory-bounded composition of Qlib's unchanged Alpha158 learning processors."""

from __future__ import annotations

import pandas as pd
from qlib.data.dataset.processor import CSZScoreNorm, DropnaLabel, Processor


class BaselineLabelProcessor(Processor):
    """Apply DropnaLabel then label CSZScoreNorm without a redundant input copy.

    Pinned Qlib normally copies the complete Alpha158 frame before this chain
    because CSZScoreNorm mutates its argument. DropnaLabel already returns a
    separate frame, so only that result needs to be mutated. The composition
    is read-only with respect to its caller, although its second stage is not.
    Both upstream processors have no-op fit methods and remain unchanged.
    """

    def __init__(self) -> None:
        self.dropna = DropnaLabel()
        self.normalize = CSZScoreNorm(fields_group="label")

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.normalize(self.dropna(df))

    def readonly(self) -> bool:
        return True

    def is_for_infer(self) -> bool:
        return False
