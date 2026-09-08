from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.no_database
RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "model_sandbox_runner.py"


def runner_module():
    spec = importlib.util.spec_from_file_location("model_sandbox_label_contract", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def label_fixture(monkeypatch):
    index = pd.MultiIndex.from_product(
        [pd.date_range("2025-01-06", periods=2), ["SH600000", "SZ000001"]],
        names=["datetime", "instrument"],
    )
    labels = pd.DataFrame({"LABEL0": [0.02, -0.01, 0.04, np.nan]}, index=index)
    calls = []
    handler = object()
    segments = {"train": ("2024-01-01", "2024-10-31"),
                "valid": ("2024-11-04", "2024-12-31"), "test": ("2025-01-06", "2025-01-07")}
    fetch_kwargs = {"fetch_orig": True}

    class DatasetH:
        def __init__(self, **kwargs):
            assert kwargs["handler"] is handler
            assert kwargs["segments"] is segments
            assert kwargs["fetch_kwargs"] is fetch_kwargs
            calls.append(("shared_handler", kwargs))

        def prepare(self, segment, *, col_set, data_key):
            calls.append(("prepare", segment, col_set, data_key))
            return state.labels

    modules = {name: ModuleType(name) for name in (
        "qlib", "qlib.data", "qlib.data.dataset", "qlib.data.dataset.handler",
    )}
    modules["qlib.data.dataset"].DatasetH = DatasetH
    modules["qlib.data.dataset.handler"].DataHandlerLP = SimpleNamespace(DK_I="infer")
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    def sampled_prepare(*args, **kwargs):
        raise AssertionError("evaluation must not construct a time-series sampler")

    dataset = SimpleNamespace(handler=handler, segments=segments, fetch_kwargs=fetch_kwargs,
                              prepare=sampled_prepare)
    state = SimpleNamespace(labels=labels, dataset=dataset, calls=calls, index=index)
    return state


def test_unsampled_labels_share_handler_keep_raw_values_and_missing_maturity(label_fixture):
    fixture = label_fixture
    original = fixture.labels.copy(deep=True)
    labels = runner_module().prepare_evaluation_labels(fixture.dataset, fixture.index)
    assert labels is fixture.labels
    pd.testing.assert_frame_equal(labels, original)
    assert fixture.calls[-1] == ("prepare", "test", "label", "infer")
    assert pd.isna(labels.iloc[-1, 0])


def test_labels_align_by_stock_and_date_in_prediction_order_not_sampler_position(label_fixture):
    fixture = label_fixture
    predictions = pd.Series([0.6, -0.3, 0.1], index=fixture.index[[3, 0, 2]], name="score")
    labels = runner_module().prepare_evaluation_labels(fixture.dataset, predictions.index)
    pd.testing.assert_frame_equal(labels, fixture.labels.iloc[[3, 0, 2]])
    assert labels.index.equals(predictions.index)
    # Metric rows may omit an immature target; the recorder label frame must retain it.
    aligned = pd.concat([predictions, labels.iloc[:, 0].rename("label")], axis=1).dropna()
    assert len(aligned) == 2 and len(labels) == 3 and pd.isna(labels.iloc[0, 0])


@pytest.mark.parametrize("kind", ["duplicate", "wrong-names", "not-multiindex"])
def test_ambiguous_prediction_indices_are_rejected_before_label_access(label_fixture, kind):
    fixture = label_fixture
    invalid = {
        "duplicate": fixture.index[[0, 0]],
        "wrong-names": fixture.index.set_names(["date", "stock"]),
        "not-multiindex": pd.Index([1, 2]),
    }[kind]
    with pytest.raises(ValueError, match="unique datetime/instrument index"):
        runner_module().prepare_evaluation_labels(fixture.dataset, invalid)
    assert not fixture.calls


@pytest.mark.parametrize("kind", ["duplicate", "wrong-names", "multiple-labels", "sampler"])
def test_label_shape_and_identity_cannot_silently_change_metrics(label_fixture, kind):
    fixture = label_fixture
    if kind == "duplicate":
        fixture.labels = fixture.labels.iloc[[0, 0]]
    elif kind == "wrong-names":
        fixture.labels = fixture.labels.rename_axis(index=["date", "stock"])
    elif kind == "multiple-labels":
        fixture.labels = fixture.labels.assign(LABEL1=100.0)
    else:
        fixture.labels = SimpleNamespace()
    with pytest.raises(ValueError, match="one label per unique datetime/instrument"):
        runner_module().prepare_evaluation_labels(fixture.dataset, fixture.index)


@pytest.mark.parametrize("missing", [("2025-01-08", "SH600000"), ("2025-01-06", "SH600001")])
def test_prediction_outside_frozen_test_labels_is_rejected(label_fixture, missing):
    outside = pd.MultiIndex.from_tuples([(pd.Timestamp(missing[0]), missing[1])],
                                       names=["datetime", "instrument"])
    with pytest.raises(ValueError, match="escaped the frozen test label index"):
        runner_module().prepare_evaluation_labels(label_fixture.dataset, outside)


@pytest.mark.parametrize("time_series", [False, True], ids=["DatasetH", "TSDatasetH"])
def test_real_qlib_test_labels_bypass_sequences_and_preserve_prepared_raw_space(
    monkeypatch, time_series,
):
    qlib_dataset = pytest.importorskip("qlib.data.dataset")
    from qlib.data.dataset.handler import DataHandlerLP

    from quant_platform.model_data_handler import (
        build_model_handler_from_prepared_data,
        prepare_model_data_from_columns,
    )

    dates = pd.bdate_range("2025-01-02", periods=8)
    index = pd.MultiIndex.from_product([dates, ["SH600000", "SZ000001", "SZ000002"]],
                                      names=["datetime", "instrument"])
    values = np.tile(np.array([0.02, -0.01, 0.15], dtype=np.float32), len(dates))
    values[-1] = np.nan
    prepared = prepare_model_data_from_columns(
        {("feature", "f0"): np.arange(len(index), dtype=np.float32),
         ("label", "LABEL0"): values.copy()},
        index, fit_start_time=str(dates[0].date()), fit_end_time=str(dates[2].date()),
    )
    handler = build_model_handler_from_prepared_data(prepared)
    segments = {"train": (str(dates[0].date()), str(dates[2].date())),
                "valid": (str(dates[3].date()), str(dates[4].date())),
                "test": (str(dates[5].date()), str(dates[-1].date()))}
    dataset = (qlib_dataset.TSDatasetH(handler=handler, segments=segments, step_len=3)
               if time_series else qlib_dataset.DatasetH(handler=handler, segments=segments))
    sampled = dataset.prepare("test", col_set="label")
    if time_series:
        assert not hasattr(sampled, "iloc")
        prediction_index = sampled.get_index()[::-1]
    else:
        prediction_index = sampled.index[::-1]
    calls = []
    fetch = handler.fetch

    def labels_only(*args, **kwargs):
        assert kwargs["col_set"] == "label" and kwargs["data_key"] == DataHandlerLP.DK_I
        calls.append((args, kwargs))
        return fetch(*args, **kwargs)

    def reject_sampler(*args, **kwargs):
        raise AssertionError("label recording may not allocate another time-series sampler")

    monkeypatch.setattr(handler, "fetch", labels_only)
    monkeypatch.setattr(handler, "setup_data", reject_sampler)
    monkeypatch.setattr(qlib_dataset, "TSDataSampler", reject_sampler)
    labels = runner_module().prepare_evaluation_labels(dataset, prediction_index)
    expected = pd.DataFrame({"LABEL0": values}, index=index).reindex(prediction_index)
    pd.testing.assert_frame_equal(labels, expected)
    assert len(calls) == 1 and calls[0][0][0] == segments["test"]
    assert dataset.handler is handler
    assert labels.index.get_level_values("datetime").min() == dates[5]
    assert labels.index.get_level_values("datetime").max() == dates[-1]
    assert labels.isna().sum().iloc[0] == 1
    learning = prepared.learn_labels["label"].reindex(prediction_index)
    assert not labels.equals(learning)


@pytest.mark.parametrize("model_class", ["GovernedGRU", "GovernedTransformer"])
def test_real_sequence_fit_predict_raw_labels_ic_and_recorder_roundtrip(
    tmp_path, monkeypatch, model_class,
):
    qlib = pytest.importorskip("qlib")
    torch = pytest.importorskip("torch")
    mlflow = pytest.importorskip("mlflow")
    from qlib.contrib.model.pytorch_general_nn import GeneralPTNN
    from qlib.data.dataset import TSDatasetH
    from qlib.workflow import R

    from quant_platform.model_data_handler import (
        build_model_handler_from_prepared_data,
        prepare_model_data_from_columns,
    )

    # Use the production architectures and Qlib trainer on entirely synthetic
    # in-memory data. No market provider or production files are accessed.
    module = runner_module()
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("GIT_PYTHON_REFRESH", "quiet")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "")
    monkeypatch.setenv("_MLFLOW_SERVER_ARTIFACT_ROOT", "")
    provider = tmp_path / "empty-provider"
    provider.mkdir()
    tracking_uri = module.initialize_model_qlib(
        qlib, output=tmp_path / "output", provider_uri=str(provider), kernels=1,
    )
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(11)
        dates = pd.bdate_range("2025-01-02", periods=48)
        index = pd.MultiIndex.from_product(
            [dates, ["SH600000", "SH600001", "SZ000001", "SZ000002"]],
            names=["datetime", "instrument"],
        )
        rng = np.random.default_rng(701)
        features = rng.normal(size=(len(index), 4)).astype(np.float32)
        raw_values = (0.03 * features[:, 0] - 0.01 * features[:, 1]).astype(np.float32)
        raw_values[-1] = np.nan
        columns = {("feature", f"F{number}"): features[:, number]
                   for number in range(features.shape[1])}
        columns[("label", "LABEL0")] = raw_values.copy()
        prepared = prepare_model_data_from_columns(
            columns, index, fit_start_time=str(dates[0].date()),
            fit_end_time=str(dates[31].date()),
        )
        handler = build_model_handler_from_prepared_data(prepared)
        segments = {"train": (str(dates[0].date()), str(dates[31].date())),
                    "valid": (str(dates[32].date()), str(dates[39].date())),
                    "test": (str(dates[40].date()), str(dates[-1].date()))}
        dataset = TSDatasetH(handler=handler, segments=segments, step_len=20)
        model = GeneralPTNN(
            n_epochs=1, lr=2e-4, metric="loss", batch_size=32, early_stop=1,
            loss="mse", weight_decay=1e-4, n_jobs=0, GPU=-1, seed=11,
            pt_model_uri=f"quant_platform.model_templates.{model_class}",
            pt_model_kwargs={"num_features": features.shape[1], "num_timesteps": 20},
        )
        checkpoint = tmp_path / "model.pt"
        evaluations = {}
        model.fit(dataset, evals_result=evaluations, save_path=str(checkpoint))
        if mlflow.active_run() is not None:
            module.end_implicit_qlib_recorder()
        assert mlflow.active_run() is None
        assert model.fitted and checkpoint.is_file()
        assert len(evaluations["train"]) == len(evaluations["valid"]) == 1
        assert np.isfinite(evaluations["train"] + evaluations["valid"]).all()

        predictions = model.predict(dataset).rename("score").sort_index()
        assert len(predictions) == 32 and np.isfinite(predictions).all()
        labels = module.prepare_evaluation_labels(dataset, predictions.index)
        expected = pd.DataFrame({"LABEL0": raw_values}, index=index).reindex(predictions.index)
        pd.testing.assert_frame_equal(labels, expected, check_exact=True)
        assert labels.index.get_level_values("datetime").min() == dates[40]
        assert labels.index.get_level_values("datetime").max() == dates[-1]
        assert labels.isna().sum().iloc[0] == 1
        assert not labels.equals(prepared.learn_labels["label"].reindex(predictions.index))

        aligned = pd.concat([predictions, labels.iloc[:, 0].rename("label")], axis=1).dropna()
        assert len(aligned) == 31
        daily_ic = aligned.groupby(level="datetime").apply(
            lambda frame: frame["score"].corr(frame["label"]), include_groups=False,
        )
        daily_rank_ic = aligned.groupby(level="datetime").apply(
            lambda frame: frame["score"].corr(frame["label"], method="spearman"),
            include_groups=False,
        )
        assert len(daily_ic) == len(daily_rank_ic) == 8
        assert np.isfinite(daily_ic).all() and np.isfinite(daily_rank_ic).all()
        with module.qlib_workflow_run(
            run_kind="model-label-fixture", run_id=model_class,
            tracking_uri=tracking_uri, dataset_identity_sha256="a" * 64,
        ) as workflow:
            recorder = workflow.get_recorder()
            recorder.save_objects(**{"pred.pkl": predictions.to_frame(), "label.pkl": labels})
            recorder.log_metrics(IC=float(daily_ic.mean()), RankIC=float(daily_rank_ic.mean()))
            pd.testing.assert_frame_equal(recorder.load_object("label.pkl"), expected,
                                          check_exact=True)
            pd.testing.assert_frame_equal(recorder.load_object("pred.pkl"), predictions.to_frame(),
                                          check_exact=True)
        assert mlflow.active_run() is None
    finally:
        R.end_exp()
        torch.set_num_threads(previous_threads)
