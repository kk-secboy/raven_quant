"""Native fixed-image check; deliberately opt-in outside the Qlib runtime."""
from __future__ import annotations

import hashlib
import os

import pytest

pytestmark = [pytest.mark.no_database, pytest.mark.skipif(
    os.environ.get("QUANTLAB_NATIVE_THREAD_TESTS") != "1",
    reason="requires the fixed native Qlib model image",
)]


@pytest.mark.parametrize("lane", ["tabular", "timeseries"])
@pytest.mark.parametrize("seed", [11, 29, 47])
def test_native_qlib_loader_zero_and_two_preserve_exact_training_and_predictions(
    tmp_path, monkeypatch, lane, seed,
):
    import numpy as np
    import pandas as pd
    import torch
    from qlib.contrib.model.pytorch_general_nn import GeneralPTNN
    from qlib.data.dataset import DatasetH, TSDatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    # Exact equality is the preregistered criterion for loader changes. Compute
    # threads remain fixed at two here, so no floating-point tolerance is used.
    torch.set_num_threads(2)
    dates = pd.bdate_range("2021-01-04", periods=80)
    index = pd.MultiIndex.from_product(
        [dates, ["SH600000", "SH600001", "SZ000001", "SZ000002"]],
        names=["datetime", "instrument"],
    )
    rng = np.random.default_rng(701)
    values = rng.normal(size=(len(index), 5)).astype("float32")
    columns = pd.MultiIndex.from_tuples(
        [("feature", f"F{number}") for number in range(4)] + [("label", "LABEL0")],
    )
    raw = pd.DataFrame(values, index=index, columns=columns)
    segments = {"train": (str(dates[0].date()), str(dates[49].date())),
                "valid": (str(dates[50].date()), str(dates[64].date())),
                "test": (str(dates[65].date()), str(dates[-1].date()))}
    fixture_module = tmp_path / "native_thread_fixture.py"
    fixture_module.write_text(
        "import torch\n"
        "class NativeTabular(torch.nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.network = torch.nn.Sequential(torch.nn.Linear(4, 8), "
        "torch.nn.ReLU(), torch.nn.Dropout(0.1), torch.nn.Linear(8, 1))\n"
        "    def forward(self, value):\n"
        "        return self.network(value)\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    outputs = []
    original_train_epoch = GeneralPTNN.train_epoch

    for workers in (0, 2):
        np.random.seed(seed)
        torch.manual_seed(seed)
        handler = DataHandlerLP(data_loader=StaticDataLoader(raw.copy()), drop_raw=True)
        dataset = (
            DatasetH(handler=handler, segments=segments) if lane == "tabular"
            else TSDatasetH(handler=handler, segments=segments, step_len=20)
        )
        model = GeneralPTNN(
            n_epochs=3, lr=2e-4, metric="loss", batch_size=32, early_stop=3,
            loss="mse", weight_decay=1e-4, n_jobs=workers, GPU=-1, seed=seed,
            pt_model_uri=("native_thread_fixture.NativeTabular" if lane == "tabular"
                          else "quant_platform.model_templates.GovernedGRU"),
            pt_model_kwargs={} if lane == "tabular" else {"num_features": 4},
        )
        batches = []

        def tracked_epoch(loader, batches=batches, model=model):
            def track():
                for data, weight in loader:
                    batches.append(hashlib.sha256(
                        data.numpy().tobytes() + weight.numpy().tobytes(),
                    ).hexdigest())
                    yield data, weight
            original_train_epoch(model, track())

        model.train_epoch = tracked_epoch
        evaluations = {}
        model.fit(dataset, evals_result=evaluations, save_path=str(tmp_path / f"{workers}.pt"))
        predictions = model.predict(dataset)
        weights = {name: value.detach().cpu().numpy().copy()
                   for name, value in model.dnn_model.state_dict().items()}
        outputs.append((batches, evaluations, predictions, weights))

    assert outputs[0][0] == outputs[1][0], "DataLoader workers changed shuffled batch order"
    assert outputs[0][1] == outputs[1][1], "DataLoader workers changed epoch losses"
    pd.testing.assert_series_equal(outputs[0][2], outputs[1][2], check_exact=True)
    for name in outputs[0][3]:
        np.testing.assert_array_equal(outputs[0][3][name], outputs[1][3][name])
