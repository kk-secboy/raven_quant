from __future__ import annotations

import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import numpy as np
import pandas as pd
import pytest

from quant_platform.model_prepared_data import (
    PreparedModelData,
    canonical_key,
    load_prepared_data,
    publish_prepared_data,
    write_prepared_data,
)

pytestmark = pytest.mark.no_database


@pytest.fixture
def contract():
    return {
        "dataset_identity": "a" * 64,
        "features": [["F0", "$close"], ["F1", "$close/Ref($close,1)-1"]],
        "label_expression": "Ref($close,-2)/Ref($close,-1)-1",
        "universe": "cn_all",
        "load_start": "2020-01-02",
        "load_end": "2020-01-04",
        "fit_start": "2020-01-02",
        "fit_end": "2020-01-03",
        "additional_factors_sha256": None,
        "processor_identity": "b" * 64,
        "runtime_identity": "c" * 64,
    }


@pytest.fixture
def prepared():
    # Unused date/instrument levels survive slicing in Qlib. They must not be
    # reconstructed from only the observed rows or silently lose their dtype.
    index = pd.MultiIndex(
        levels=[pd.date_range("2020-01-02", periods=4),
                pd.Index(["SH001", "SZ002", "UNUSED"])],
        codes=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]],
        names=["datetime", "instrument"],
    )
    # Distinct NaN payloads and negative zero detect lossy serialisation that
    # numerical equality (including assert_array_equal) would otherwise miss.
    values32 = np.array(
        [0x3F800000, 0x7FC00017, 0x80000000, 0x40000000, 0x7FC00024, 0x3F000000],
        dtype="uint32",
    ).view("float32")
    values64 = np.array([1.0 / 3, np.nan, -0.0, 17.0, -2.125, 1e-13], dtype="float64")
    labels = pd.DataFrame({("label", "LABEL0"): values32.copy(),
                           ("label", "LABEL1"): values64.copy()}, index=index, copy=False)
    learn = labels.iloc[[0, 2, 3, 5]].copy()
    return PreparedModelData(
        index=index,
        features={("feature", "F0"): values32, ("feature", "F1"): values64},
        infer_labels=labels,
        learn_labels=learn,
    )


def _assert_index_exact(expected, actual):
    pd.testing.assert_index_equal(expected, actual, exact=True)
    assert expected.names == actual.names
    for expected_level, actual_level in zip(expected.levels, actual.levels, strict=True):
        pd.testing.assert_index_equal(expected_level, actual_level, exact=True)
    for expected_codes, actual_codes in zip(expected.codes, actual.codes, strict=True):
        assert expected_codes.dtype == actual_codes.dtype
        assert expected_codes.tobytes() == actual_codes.tobytes()


def _assert_array_bytes(expected, actual):
    assert expected.dtype == actual.dtype
    assert expected.shape == actual.shape
    assert expected.tobytes() == actual.tobytes()


def _assert_prepared_exact(expected, actual):
    _assert_index_exact(expected.index, actual.index)
    assert list(actual.features) == list(expected.features)
    for key in expected.features:
        _assert_array_bytes(expected.features[key], actual.features[key])
    for label_kind in ("infer_labels", "learn_labels"):
        expected_labels, actual_labels = getattr(expected, label_kind), getattr(actual, label_kind)
        _assert_index_exact(expected_labels.index, actual_labels.index)
        pd.testing.assert_index_equal(expected_labels.columns, actual_labels.columns, exact=True)
        for key in expected_labels.columns:
            _assert_array_bytes(expected_labels[key].to_numpy(), actual_labels[key].to_numpy())


def _artifact_files(directory):
    return sorted(directory.rglob("*.npy"))


def _rewrite_manifest(directory, mutate):
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_identity_is_canonical_for_mappings_but_preserves_feature_order(contract):
    reordered = dict(reversed(list(contract.items())))
    assert canonical_key(contract) == canonical_key(reordered)
    assert len(canonical_key(contract)) == 64
    int(canonical_key(contract), 16)
    reordered["features"] = list(reversed(reordered["features"]))
    assert canonical_key(contract) != canonical_key(reordered)


@pytest.mark.parametrize("field,replacement", [
    ("dataset_identity", "d" * 64),
    ("features", [["F0", "$open"], ["F1", "$close/Ref($close,1)-1"]]),
    ("label_expression", "Ref($close,-3)/Ref($close,-1)-1"),
    ("universe", "csi300"),
    ("load_start", "2020-01-01"),
    ("load_end", "2020-01-05"),
    ("fit_start", "2020-01-01"),
    ("fit_end", "2020-01-02"),
    ("additional_factors_sha256", "e" * 64),
    ("processor_identity", "f" * 64),
    ("runtime_identity", "0" * 64),
])
def test_changed_preparation_inputs_cannot_load_old_artifact(
    tmp_path, prepared, contract, field, replacement,
):
    directory = tmp_path / "prepared"
    write_prepared_data(directory, contract=contract, data=prepared)
    changed = copy.deepcopy(contract)
    changed[field] = replacement
    assert canonical_key(changed) != canonical_key(contract)
    with pytest.raises(ValueError):
        load_prepared_data(directory, expected_contract=changed)


@pytest.mark.parametrize("names", [["datetime", "instrument"], [None, "security"]])
def test_roundtrip_preserves_bytes_dtypes_order_and_unused_index_levels(
    tmp_path, prepared, contract, names,
):
    prepared.index.names = names
    prepared.infer_labels.index.names = names
    prepared.learn_labels.index.names = names
    directory = tmp_path / "prepared"
    write_prepared_data(directory, contract=contract, data=prepared)
    restored = load_prepared_data(directory, expected_contract=contract)
    _assert_prepared_exact(prepared, restored)


def test_loaded_arrays_cannot_mutate_shared_artifacts(tmp_path, prepared, contract):
    directory = tmp_path / "prepared"
    write_prepared_data(directory, contract=contract, data=prepared)
    arrays = _artifact_files(directory)
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in arrays}
    restored = load_prepared_data(directory, expected_contract=contract)
    for values in restored.features.values():
        assert not values.flags.writeable
        with pytest.raises(ValueError):
            values[0] = 123
    for labels in (restored.infer_labels, restored.learn_labels):
        for key in labels:
            assert not labels[key].to_numpy().flags.writeable
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in arrays}
    assert before == after


def test_existing_directory_cannot_be_overwritten(tmp_path, prepared, contract):
    directory = tmp_path / "prepared"
    write_prepared_data(directory, contract=contract, data=prepared)
    before = {str(path.relative_to(directory)): path.read_bytes()
              for path in directory.rglob("*") if path.is_file()}
    with pytest.raises((FileExistsError, ValueError)):
        write_prepared_data(directory, contract=contract, data=prepared)
    after = {str(path.relative_to(directory)): path.read_bytes()
             for path in directory.rglob("*") if path.is_file()}
    assert before == after


@pytest.mark.parametrize("damage", ["flip_byte", "truncate", "remove"])
def test_damaged_array_is_rejected_before_data_is_returned(
    tmp_path, prepared, contract, damage,
):
    directory = tmp_path / "prepared"
    write_prepared_data(directory, contract=contract, data=prepared)
    path = _artifact_files(directory)[0]
    if damage == "remove":
        path.unlink()
    elif damage == "truncate":
        path.write_bytes(path.read_bytes()[:-1])
    else:
        damaged = bytearray(path.read_bytes())
        damaged[-1] ^= 1
        path.write_bytes(damaged)
    with pytest.raises((ValueError, FileNotFoundError)):
        load_prepared_data(directory, expected_contract=contract)


def test_two_publishers_converge_on_one_complete_immutable_entry(tmp_path, prepared, contract):
    stages = [tmp_path / "stage_a", tmp_path / "stage_b"]
    for stage in stages:
        write_prepared_data(stage, contract=contract, data=prepared)
    cache_root = tmp_path / "cache"
    barrier = Barrier(2)

    def publish(stage):
        barrier.wait(timeout=10)
        return publish_prepared_data(stage, cache_root, contract=contract)

    with ThreadPoolExecutor(max_workers=2) as executor:
        destinations = list(executor.map(publish, stages))
    assert destinations[0] == destinations[1] == cache_root / canonical_key(contract)
    restored = load_prepared_data(destinations[0], expected_contract=contract)
    _assert_prepared_exact(prepared, restored)


def test_corrupt_published_winner_is_rejected_and_not_replaced(tmp_path, prepared, contract):
    cache_root = tmp_path / "cache"
    first, second = tmp_path / "first", tmp_path / "second"
    write_prepared_data(first, contract=contract, data=prepared)
    winner = publish_prepared_data(first, cache_root, contract=contract)
    victim = _artifact_files(winner)[0]
    victim.write_bytes(b"corrupt winner")
    write_prepared_data(second, contract=contract, data=prepared)
    with pytest.raises(ValueError):
        publish_prepared_data(second, cache_root, contract=contract)
    assert victim.read_bytes() == b"corrupt winner"


def test_array_symlink_is_rejected_even_when_target_bytes_are_valid(tmp_path, prepared, contract):
    directory = tmp_path / "prepared"
    write_prepared_data(directory, contract=contract, data=prepared)
    victim = _artifact_files(directory)[0]
    outside = tmp_path / "outside.npy"
    outside.write_bytes(victim.read_bytes())
    victim.unlink()
    try:
        victim.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")
    with pytest.raises(ValueError):
        load_prepared_data(directory, expected_contract=contract)


def test_object_dtype_is_not_accepted_as_feature_data(tmp_path, prepared, contract):
    prepared.features[("feature", "F0")] = np.array(["unsafe"] * 6, dtype=object)
    with pytest.raises((TypeError, ValueError)):
        write_prepared_data(tmp_path / "prepared", contract=contract, data=prepared)


def test_missing_commit_manifest_never_returns_partial_data(tmp_path, prepared, contract):
    directory = tmp_path / "prepared"
    write_prepared_data(directory, contract=contract, data=prepared)
    (directory / "manifest.json").unlink()
    assert _artifact_files(directory)
    with pytest.raises(ValueError):
        load_prepared_data(directory, expected_contract=contract)
    with pytest.raises(ValueError):
        publish_prepared_data(directory, tmp_path / "cache", contract=contract)


def test_expected_manifest_pin_rejects_changed_or_wrong_manifest(tmp_path, prepared, contract):
    directory = tmp_path / "prepared"
    write_prepared_data(directory, contract=contract, data=prepared)
    path = directory / "manifest.json"
    pin = hashlib.sha256(path.read_bytes()).hexdigest()
    _assert_prepared_exact(prepared, load_prepared_data(
        directory, expected_contract=contract, expected_manifest_sha256=pin,
    ))
    with pytest.raises(ValueError):
        load_prepared_data(directory, expected_contract=contract,
                           expected_manifest_sha256="0" * 64)
    # Even a whitespace-only change must invalidate the exact manifest pin.
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        load_prepared_data(directory, expected_contract=contract, expected_manifest_sha256=pin)


@pytest.mark.parametrize("unsafe_path", [
    "../outside.npy", "/outside.npy", "C:/outside.npy", "..\\outside.npy",
    "nested/feature.npy", "feature-0000.npy:stream",
])
def test_manifest_cannot_redirect_feature_reads(tmp_path, prepared, contract, unsafe_path):
    directory = tmp_path / "prepared"
    write_prepared_data(directory, contract=contract, data=prepared)

    def redirect(manifest):
        original = manifest["features"][0]["file"]
        manifest["features"][0]["file"] = unsafe_path
        manifest["files"][unsafe_path] = manifest["files"].pop(original)

    _rewrite_manifest(directory, redirect)
    with pytest.raises(ValueError):
        load_prepared_data(directory, expected_contract=contract)


def test_object_npy_cannot_be_loaded_even_with_matching_file_digest(tmp_path, prepared, contract):
    directory = tmp_path / "prepared"
    write_prepared_data(directory, contract=contract, data=prepared)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    filename = manifest["features"][0]["file"]
    path = directory / filename
    values = np.array([{"unexpected": "object"}] * len(prepared.index), dtype=object)
    np.save(path, values, allow_pickle=True)

    def replace_descriptor(manifest):
        descriptor = manifest["files"][filename]
        descriptor["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        descriptor["size_bytes"] = path.stat().st_size
        descriptor["shape"] = list(values.shape)
        descriptor["dtype"] = values.dtype.str

    _rewrite_manifest(directory, replace_descriptor)
    with pytest.raises(ValueError):
        load_prepared_data(directory, expected_contract=contract)


def test_label_indexes_cannot_change_the_governed_row_universe(tmp_path, prepared, contract):
    prepared.infer_labels = prepared.infer_labels.iloc[:-1]
    with pytest.raises(ValueError):
        write_prepared_data(tmp_path / "prepared", contract=contract, data=prepared)


def test_learn_labels_must_be_subset_of_inference_rows(tmp_path, prepared, contract):
    bad = prepared.learn_labels.copy()
    changed = list(bad.index)
    changed[0] = (pd.Timestamp("2020-01-05"), "SH001")
    bad.index = pd.MultiIndex.from_tuples(changed, names=bad.index.names)
    prepared.learn_labels = bad.sort_index()
    with pytest.raises(ValueError):
        write_prepared_data(tmp_path / "prepared", contract=contract, data=prepared)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), Path("not-json")])
def test_non_canonical_contract_values_are_rejected(contract, value):
    contract["invalid"] = value
    with pytest.raises((TypeError, ValueError)):
        canonical_key(contract)


@pytest.mark.parametrize("field,replacement", [("shape", [999]), ("dtype", "<i2")])
def test_manifest_array_metadata_must_match_verified_bytes(
    tmp_path, prepared, contract, field, replacement,
):
    directory = tmp_path / "prepared"
    write_prepared_data(directory, contract=contract, data=prepared)

    def change_metadata(manifest):
        name = manifest["features"][0]["file"]
        manifest["files"][name][field] = replacement

    _rewrite_manifest(directory, change_metadata)
    with pytest.raises(ValueError):
        load_prepared_data(directory, expected_contract=contract)


def test_failed_array_write_never_publishes_a_complete_manifest(
    tmp_path, prepared, contract, monkeypatch,
):
    real_save = np.save
    writes = 0

    def fail_during_write(*args, **kwargs):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("simulated disk write failure")
        return real_save(*args, **kwargs)

    monkeypatch.setattr(np, "save", fail_during_write)
    directory = tmp_path / "incomplete"
    with pytest.raises(OSError, match="simulated disk write failure"):
        write_prepared_data(directory, contract=contract, data=prepared)
    assert not (directory / "manifest.json").exists()
    with pytest.raises(ValueError):
        load_prepared_data(directory, expected_contract=contract)


@pytest.mark.parametrize("target", ["array_dtype", "label_column_kind"])
def test_malformed_manifest_values_raise_a_validation_error(tmp_path, prepared, contract, target):
    directory = tmp_path / "prepared"
    write_prepared_data(directory, contract=contract, data=prepared)

    def malformed(manifest):
        if target == "array_dtype":
            filename = manifest["features"][0]["file"]
            manifest["files"][filename]["dtype"] = ["<f4"]
        else:
            manifest["infer_labels"]["column_index_kind"] = ["multiindex"]

    _rewrite_manifest(directory, malformed)
    with pytest.raises(ValueError):
        load_prepared_data(directory, expected_contract=contract)


def test_duplicate_manifest_json_keys_are_not_silently_accepted(tmp_path, prepared, contract):
    directory = tmp_path / "prepared"
    write_prepared_data(directory, contract=contract, data=prepared)
    path = directory / "manifest.json"
    text = path.read_text(encoding="utf-8")
    # Keep both duplicated values correct: a normal last-key-wins parser would
    # accept this, so the failure specifically proves strict JSON parsing.
    duplicate = '{"key":' + json.dumps(canonical_key(contract)) + "," + text[1:]
    path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(ValueError):
        load_prepared_data(directory, expected_contract=contract)


def test_cyclic_contract_raises_a_validation_error(contract):
    contract["self"] = contract
    with pytest.raises(ValueError):
        canonical_key(contract)


@pytest.mark.parametrize("index_field", ["datetime_level", "datetime_codes"])
def test_object_index_arrays_are_rejected_with_matching_checksums(
    tmp_path, prepared, contract, index_field,
):
    directory = tmp_path / "prepared"
    write_prepared_data(directory, contract=contract, data=prepared)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    filename = manifest["index"][index_field]
    path = directory / filename
    values = np.array(["object"] * manifest["files"][filename]["shape"][0], dtype=object)
    np.save(path, values, allow_pickle=True)

    def update_metadata(manifest):
        manifest["files"][filename].update({
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size, "dtype": values.dtype.str,
        })

    _rewrite_manifest(directory, update_metadata)
    with pytest.raises(ValueError):
        load_prepared_data(directory, expected_contract=contract)


@pytest.mark.parametrize("invalid_code", [-2, 999])
def test_index_codes_cannot_escape_bound_levels_even_with_matching_checksums(
    tmp_path, prepared, contract, invalid_code,
):
    directory = tmp_path / "prepared"
    write_prepared_data(directory, contract=contract, data=prepared)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    filename = manifest["index"]["instrument_codes"]
    path = directory / filename
    codes = np.load(path, allow_pickle=False)
    codes[0] = invalid_code
    np.save(path, codes, allow_pickle=False)

    def update_metadata(manifest):
        manifest["files"][filename].update({
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        })

    _rewrite_manifest(directory, update_metadata)
    with pytest.raises(ValueError):
        load_prepared_data(directory, expected_contract=contract)


def test_index_code_file_bindings_cannot_be_swapped(tmp_path, prepared, contract):
    directory = tmp_path / "prepared"
    write_prepared_data(directory, contract=contract, data=prepared)

    def swap_files(manifest):
        index = manifest["index"]
        index["datetime_codes"], index["instrument_codes"] = (
            index["instrument_codes"], index["datetime_codes"],
        )

    _rewrite_manifest(directory, swap_files)
    with pytest.raises(ValueError):
        load_prepared_data(directory, expected_contract=contract)
