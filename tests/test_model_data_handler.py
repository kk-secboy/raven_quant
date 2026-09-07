from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("qlib")

from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.loader import StaticDataLoader

from quant_platform.model_data_handler import (
    build_model_handler_from_columns,
    build_model_handler_from_prepared_data,
    load_memory_bounded_model_handler,
    prepare_memory_bounded_model_data,
    prepare_model_data_from_columns,
)
from quant_platform.model_prepared_data import (
    load_prepared_data,
    manifest_sha256,
    write_prepared_data,
)

pytestmark = pytest.mark.no_database


def _raw(mixed=False):
    rng = np.random.default_rng(41)
    index = pd.MultiIndex.from_product(
        [pd.bdate_range("2020-01-02", periods=60), ["SH001", "SH002", "SZ003", "SZ004"]],
        names=["datetime", "instrument"],
    )
    values = rng.normal(size=(len(index), 19)).astype("float32")
    values[2, 0] = np.nan
    values[:, 3] = np.nan
    values[:, 8] = 0
    values[-20:, 10] *= 100
    columns = {("feature", f"F{i:02d}"): values[:, i] for i in range(19)}
    if mixed:
        columns[("feature", "F18")] = rng.normal(size=len(index)).astype("float64")
    labels = rng.normal(size=len(index)).astype("float32")
    labels[::9] = np.nan
    columns[("label", "LABEL0")] = labels
    return pd.DataFrame(columns, index=index, copy=False)


def _original(raw):
    return DataHandlerLP(
        data_loader=StaticDataLoader(raw), drop_raw=True,
        process_type=DataHandlerLP.PTYPE_A,
        infer_processors=[
            {"class": "RobustZScoreNorm", "kwargs": {
                "fields_group": "feature", "clip_outlier": True,
                "fit_start_time": "2020-01-02", "fit_end_time": "2020-02-14",
            }},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ],
        learn_processors=[
            {"class": "DropnaLabel"},
            {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
        ],
    )


def _cached_handler(data, directory, *, contract=None):
    contract = contract or {"fixture": "handler-equivalence", "processor": "original-qlib"}
    write_prepared_data(directory, contract=contract, data=data)
    restored = load_prepared_data(
        directory, expected_contract=contract, expected_manifest_sha256=manifest_sha256(directory),
    )
    return build_model_handler_from_prepared_data(restored)


def _bounded(raw, *, cache_dir=None, fit_end="2020-02-14"):
    kwargs = dict(
        fit_start_time="2020-01-02", fit_end_time=fit_end,
    )
    columns = {key: raw[key].to_numpy(copy=True) for key in raw.columns}
    if cache_dir is not None:
        return _cached_handler(
            prepare_model_data_from_columns(columns, raw.index, **kwargs), cache_dir,
            contract={"fixture": "handler-equivalence", "fit_end": fit_end},
        )
    return build_model_handler_from_columns(columns, raw.index, **kwargs)


def _load_handler(*, cache_dir=None, **kwargs):
    if cache_dir is not None:
        return _cached_handler(prepare_memory_bounded_model_data(**kwargs), cache_dir)
    return load_memory_bounded_model_handler(**kwargs)


def _assert_frame_bytes(expected, actual):
    pd.testing.assert_frame_equal(expected, actual, check_exact=True)
    for position in range(len(expected.columns)):
        expected_values = expected.iloc[:, position].to_numpy()
        actual_values = actual.iloc[:, position].to_numpy()
        assert expected_values.dtype == actual_values.dtype
        assert expected_values.tobytes() == actual_values.tobytes()


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_original_qlib_exact_values_indexes_dtypes_and_fetch_selection(tmp_path, mixed, cached):
    raw = _raw(mixed)
    original = _original(raw.copy())
    bounded = _bounded(raw, cache_dir=tmp_path / "prepared" if cached else None)
    for data_key in (DataHandlerLP.DK_I, DataHandlerLP.DK_L):
        for col_set in ("feature", "label", ["feature", "label"], ["label", "feature"],
                        DataHandlerLP.CS_RAW, DataHandlerLP.CS_ALL):
            for selector, level in ((slice(None), "datetime"),
                                    (slice("2020-01-10", "2020-02-21"), "datetime"),
                                    ("SH002", "instrument")):
                _assert_frame_bytes(
                    original.fetch(selector, level=level, col_set=col_set, data_key=data_key),
                    bounded.fetch(selector, level=level, col_set=col_set, data_key=data_key),
                )
        assert original.get_cols(data_key=data_key) == bounded.get_cols(data_key=data_key)
    with pytest.raises(AttributeError, match="drop_raw"):
        bounded.fetch(data_key=DataHandlerLP.DK_R)


@pytest.mark.parametrize("cached", [False, True])
def test_loading_batches_keep_full_expression_dates_and_instruments(tmp_path, monkeypatch, cached):
    from qlib.data.dataset.loader import QlibDataLoader

    raw = _raw()
    requests = []

    def load(loader, instruments, start_time, end_time):
        requests.append((instruments, start_time, end_time, loader.fields))
        selected = [
            (group, name) for group, (_, names) in loader.fields.items() for name in names
        ]
        return raw[selected].copy()

    monkeypatch.setattr(QlibDataLoader, "load", load)
    handler = _load_handler(
        cache_dir=tmp_path / "prepared" if cached else None,
        features={f"F{i:02d}": f"Ref($close,{i})" for i in range(19)},
        label_expression="Ref($close,-2)/Ref($close,-1)-1", instruments="cn_all",
        start_time="2020-01-02", end_time="2020-03-25", fit_end_time="2020-02-14",
    )
    assert len(requests) == 3
    assert all(row[:3] == ("cn_all", "2020-01-02", "2020-03-25") for row in requests)
    assert [len(row[3]["feature"][0]) for row in requests] == [8, 8, 3]
    for key in (DataHandlerLP.DK_I, DataHandlerLP.DK_L):
        _assert_frame_bytes(
            _original(raw.copy()).fetch(data_key=key, col_set=DataHandlerLP.CS_RAW),
            handler.fetch(data_key=key, col_set=DataHandlerLP.CS_RAW),
        )


@pytest.mark.parametrize("cached", [False, True])
def test_deterministic_ridge_predictions_remain_exact(tmp_path, cached):
    raw = _raw(mixed=True)
    original = _original(raw.copy())
    bounded = _bounded(raw, cache_dir=tmp_path / "prepared" if cached else None)
    predictions = []
    for handler in (original, bounded):
        train = handler.fetch(slice("2020-01-02", "2020-02-14"),
                              col_set=DataHandlerLP.CS_RAW, data_key=DataHandlerLP.DK_L)
        x, y = train["feature"].to_numpy(), train["label"].to_numpy().ravel()
        coefficients = np.linalg.solve(x.T @ x + np.eye(x.shape[1]), x.T @ y)
        test = handler.fetch(slice("2020-02-17", "2020-03-25"), col_set="feature")
        predictions.append(test.to_numpy() @ coefficients)
    np.testing.assert_array_equal(*predictions)
    assert predictions[0].tobytes() == predictions[1].tobytes()


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_timeseries_train_valid_test_samples_remain_exact(tmp_path, mixed, cached):
    from qlib.data.dataset import TSDatasetH

    raw = _raw(mixed)
    segments = {"train": ("2020-01-02", "2020-02-14"),
                "valid": ("2020-02-17", "2020-03-06"),
                "test": ("2020-03-09", "2020-03-25")}
    original, bounded = [
        TSDatasetH(handler=handler, segments=segments, step_len=20)
        for handler in (_original(raw.copy()), _bounded(
            raw, cache_dir=tmp_path / "prepared" if cached else None,
        ))
    ]
    assert original.cal == bounded.cal
    for segment in segments:
        for data_key in (DataHandlerLP.DK_I, DataHandlerLP.DK_L):
            expected = original.prepare(segment, col_set=["feature", "label"], data_key=data_key)
            actual = bounded.prepare(segment, col_set=["feature", "label"], data_key=data_key)
            pd.testing.assert_index_equal(expected.get_index(), actual.get_index())
            assert len(expected) == len(actual)
            assert expected.data_arr.dtype == actual.data_arr.dtype
            np.testing.assert_array_equal(expected.data_arr, actual.data_arr)
            assert expected.data_arr.tobytes() == actual.data_arr.tobytes()
            expected_samples = expected[list(range(len(expected)))]
            actual_samples = actual[list(range(len(actual)))]
            np.testing.assert_array_equal(expected_samples, actual_samples)
            assert expected_samples.tobytes() == actual_samples.tobytes()


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("additional", [False, True])
def test_real_qlib_expression_loader_and_additional_factor_equivalence(
    tmp_path, additional, cached,
):
    import qlib
    from qlib.data.dataset.loader import NestedDataLoader, QlibDataLoader

    calendar = pd.bdate_range("2020-01-02", periods=100)
    (tmp_path / "calendars").mkdir()
    (tmp_path / "instruments").mkdir()
    (tmp_path / "calendars" / "day.txt").write_text(
        "\n".join(calendar.strftime("%Y-%m-%d")), encoding="utf-8"
    )
    symbols = ["SH000001", "SH000002", "SZ000003", "SZ000004"]
    (tmp_path / "instruments" / "cn_all.txt").write_text(
        "\n".join(f"{symbol}\t{calendar[30 if i == 3 else 0]:%Y-%m-%d}"
                  f"\t{calendar[-1]:%Y-%m-%d}" for i, symbol in enumerate(symbols)),
        encoding="utf-8",
    )
    for number, symbol in enumerate(symbols):
        folder = tmp_path / "features" / symbol.lower()
        folder.mkdir(parents=True)
        values = (10 + number + np.arange(100) * 0.03 + np.sin(np.arange(100))).astype("float32")
        np.concatenate([np.array([0], dtype="float32"), values]).tofile(folder / "close.day.bin")
    qlib.init(provider_uri=str(tmp_path), region="cn", kernels=1,
              expression_cache=None, dataset_cache=None)
    features = {f"F{i:02d}": f"$close/Ref($close,{i + 1})-1" for i in range(19)}
    label_expression = "Ref($close,-2)/Ref($close,-1)-1"
    start, fit_end, end = [calendar[i].strftime("%Y-%m-%d") for i in (20, 70, 90)]
    loader = QlibDataLoader(config={
        "feature": [list(features.values()), list(features)],
        "label": [[label_expression], ["LABEL0"]],
    })
    additional_path = None
    if additional:
        raw = loader.load("cn_all", start, end)
        # Sparse float64 additions also replace one existing base-feature column,
        # exercising NestedDataLoader's left join, dtype and sorting semantics.
        extra = pd.DataFrame({
            "EXTRA": np.cos(np.arange(len(raw))),
            "F03": np.arange(len(raw), dtype="float64") / 17,
        }, index=raw.index).iloc[::2]
        additional_path = tmp_path / "additional.parquet"
        extra.to_parquet(additional_path)
        loader = NestedDataLoader([
            loader, StaticDataLoader(config={"feature": pd.read_parquet(additional_path)}),
        ])
    raw = loader.load("cn_all", start, end)
    original = DataHandlerLP(
        data_loader=StaticDataLoader(raw), drop_raw=True, process_type=DataHandlerLP.PTYPE_A,
        infer_processors=[
            {"class": "RobustZScoreNorm", "kwargs": {
                "fields_group": "feature", "clip_outlier": True,
                "fit_start_time": start, "fit_end_time": fit_end,
            }},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ],
        learn_processors=[{"class": "DropnaLabel"},
                          {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}}],
    )
    bounded = _load_handler(
        cache_dir=tmp_path / "prepared" if cached else None,
        features=features, label_expression=label_expression, instruments="cn_all",
        start_time=start, end_time=end, fit_end_time=fit_end,
        additional_factors_path=additional_path,
    )
    for data_key in (DataHandlerLP.DK_I, DataHandlerLP.DK_L):
        _assert_frame_bytes(
            original.fetch(data_key=data_key, col_set=DataHandlerLP.CS_RAW),
            bounded.fetch(data_key=data_key, col_set=DataHandlerLP.CS_RAW),
        )


def test_cached_handler_does_not_recompute_expressions_or_fit_processors(tmp_path, monkeypatch):
    from qlib.data.dataset.loader import QlibDataLoader
    from qlib.data.dataset.processor import CSZScoreNorm, RobustZScoreNorm

    raw = _raw(mixed=True)
    contract = {"fixture": "warm-only", "fit_end": "2020-02-14"}
    data = prepare_model_data_from_columns(
        {key: raw[key].to_numpy(copy=True) for key in raw.columns}, raw.index,
        fit_start_time="2020-01-02", fit_end_time="2020-02-14",
    )
    directory = tmp_path / "prepared"
    write_prepared_data(directory, contract=contract, data=data)
    original = _original(raw.copy())

    def unexpected_computation(*args, **kwargs):
        raise AssertionError("warm prepared-data load repeated Qlib preparation")

    monkeypatch.setattr(QlibDataLoader, "load", unexpected_computation)
    monkeypatch.setattr(RobustZScoreNorm, "fit", unexpected_computation)
    monkeypatch.setattr(CSZScoreNorm, "__call__", unexpected_computation)
    for _ in range(2):
        cached = build_model_handler_from_prepared_data(load_prepared_data(
            directory, expected_contract=contract,
        ))
        for key in (DataHandlerLP.DK_I, DataHandlerLP.DK_L):
            _assert_frame_bytes(
                original.fetch(data_key=key, col_set=DataHandlerLP.CS_RAW),
                cached.fetch(data_key=key, col_set=DataHandlerLP.CS_RAW),
            )


def test_normalization_uses_its_own_train_window_after_roundtrip(tmp_path):
    raw = _raw(mixed=True)
    early = _bounded(raw, cache_dir=tmp_path / "early", fit_end="2020-01-15")
    regular = _bounded(raw, cache_dir=tmp_path / "regular", fit_end="2020-02-14")
    expected_early = _bounded(raw, fit_end="2020-01-15")
    _assert_frame_bytes(expected_early.fetch(col_set="feature"), early.fetch(col_set="feature"))
    # A regression that reused the first window would still be deterministic;
    # establish that these intentionally different fits produce different data.
    assert early.fetch(col_set="feature").to_numpy().tobytes() != (
        regular.fetch(col_set="feature").to_numpy().tobytes()
    )
    # Label normalization stays over each complete date cross section, independent
    # of feature fit windows and the later model train/valid/test segment fetch.
    _assert_frame_bytes(
        early.fetch(col_set="label", data_key=DataHandlerLP.DK_L),
        regular.fetch(col_set="label", data_key=DataHandlerLP.DK_L),
    )


def test_dataset_segments_preserve_complete_cross_section_labels_after_cache_load(tmp_path):
    from qlib.data.dataset import DatasetH

    raw = _raw(mixed=True)
    segments = {"train": ("2020-01-02", "2020-02-14"),
                "valid": ("2020-02-17", "2020-03-06"),
                "test": ("2020-03-09", "2020-03-25")}
    reference, cached = [
        DatasetH(handler=handler, segments=segments)
        for handler in (_original(raw.copy()), _bounded(raw, cache_dir=tmp_path / "prepared"))
    ]
    for segment in segments:
        for data_key in (DataHandlerLP.DK_I, DataHandlerLP.DK_L):
            for cols in ("label", "feature", ["feature", "label"]):
                _assert_frame_bytes(
                    reference.prepare(segment, col_set=cols, data_key=data_key),
                    cached.prepare(segment, col_set=cols, data_key=data_key),
                )
