from __future__ import annotations

import copy
import shutil

import pytest

from quant_platform.model_data_request import prepared_data_request
from quant_platform.model_prepared_data import canonical_key

pytestmark = pytest.mark.no_database


@pytest.fixture
def provider(tmp_path):
    root = tmp_path / "provider"
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    (root / "calendars" / "day.txt").write_text(
        "2020-01-02\n2020-01-03\n2020-01-06\n2020-01-07\n"
        "2020-01-08\n2020-01-09\n2020-01-10\n", encoding="utf-8",
    )
    (root / "instruments" / "cn_all.txt").write_text(
        "SH600000\t2020-01-02\t2020-01-07\nSZ000001\t2020-01-02\t2020-01-07\n",
        encoding="utf-8",
    )
    return root


@pytest.fixture
def request_inputs(provider):
    return {
        "manifest": {
            "dataset_identity_sha256": "a" * 64,
            "feature_set": {"features": {"F0": "$close", "F1": "Ref($close,1)"}},
            "periods": {
                "train_start": "2020-01-02", "train_end": "2020-01-03",
                "valid_start": "2020-01-06", "valid_end": "2020-01-06",
                "test_start": "2020-01-07", "test_end": "2020-01-07",
            },
            "prediction_segment": "valid", "universe": "cn_all",
            "seed": 11, "candidate_id": "first-model", "model_type": "ridge",
            "model_engine": "qlib", "final_oos_opened": False,
        },
        "provider": provider,
        "label_contract": {
            "label_expression": "Ref($close,-2)/Ref($close,-1)-1", "horizon_sessions": 1,
            "purge_sessions": 0, "embargo_sessions": 0,
        },
        "producer_identity": {
            "handler_sha256": "b" * 64, "runtime_image_digest": "sha256:" + "c" * 64,
            "qlib_commit": "fixed-commit", "numpy_version": "fixed-numpy",
        },
    }


def test_different_models_seeds_and_jobs_share_identical_prepared_inputs(request_inputs):
    expected = prepared_data_request(**request_inputs)
    for model, seed in (("lightgbm", 29), ("gru", 47), ("transformer", 11)):
        changed = copy.deepcopy(request_inputs)
        changed["manifest"].update({
            "seed": seed, "model_type": model, "candidate_id": f"candidate-{model}",
            "model_engine": "pytorch" if model in {"gru", "transformer"} else "qlib",
            "job_id": "another-job", "attempt_id": "another-attempt",
        })
        assert prepared_data_request(**changed) == expected
        assert canonical_key(prepared_data_request(**changed)) == canonical_key(expected)


@pytest.mark.parametrize("date_field,new_date", [
    ("train_start", "2020-01-03"),
    ("train_end", "2020-01-02"),
    ("valid_end", "2020-01-07"),
])
def test_preparation_window_changes_require_a_different_artifact(
    request_inputs, date_field, new_date,
):
    expected = prepared_data_request(**request_inputs)
    request_inputs["manifest"]["periods"][date_field] = new_date
    if date_field == "valid_end":
        request_inputs["manifest"]["periods"].update(
            test_start="2020-01-08", test_end="2020-01-08",
        )
    assert canonical_key(prepared_data_request(**request_inputs)) != canonical_key(expected)


def test_feature_order_and_expressions_are_part_of_data_identity(request_inputs):
    expected = prepared_data_request(**request_inputs)
    original = request_inputs["manifest"]["feature_set"]["features"]
    request_inputs["manifest"]["feature_set"]["features"] = dict(reversed(list(original.items())))
    reordered = prepared_data_request(**request_inputs)
    assert canonical_key(reordered) != canonical_key(expected)
    request_inputs["manifest"]["feature_set"]["features"] = {**original, "F1": "Ref($open,1)"}
    changed_expression = prepared_data_request(**request_inputs)
    assert canonical_key(changed_expression) != canonical_key(expected)


@pytest.mark.parametrize("physical_file,extra", [
    ("calendars/day.txt", "2020-01-13\n"),
    ("instruments/cn_all.txt", "SH600001\t2020-01-02\t2020-01-07\n"),
])
def test_physical_calendar_and_market_membership_cannot_alias_old_cache(
    request_inputs, physical_file, extra,
):
    expected = prepared_data_request(**request_inputs)
    path = request_inputs["provider"] / physical_file
    with path.open("a", encoding="utf-8") as stream:
        stream.write(extra)
    changed = prepared_data_request(**request_inputs)
    assert changed["dataset_identity_sha256"] == expected["dataset_identity_sha256"]
    assert changed["load_end"] == expected["load_end"]
    assert canonical_key(changed) != canonical_key(expected)


def test_relocated_identical_provider_does_not_fragment_cache(tmp_path, request_inputs):
    expected = prepared_data_request(**request_inputs)
    relocated = tmp_path / "relocated"
    shutil.copytree(request_inputs["provider"], relocated)
    request_inputs["provider"] = relocated
    assert prepared_data_request(**request_inputs) == expected


def test_additional_factor_bytes_bind_identity_including_sparse_override_artifacts(
    tmp_path, request_inputs,
):
    without = prepared_data_request(**request_inputs)
    first = tmp_path / "additional.parquet"
    first.write_bytes(b"first frozen factor artifact")
    request_inputs["additional_factors"] = first
    with_factor = prepared_data_request(**request_inputs)
    assert canonical_key(with_factor) != canonical_key(without)
    relocated = tmp_path / "same-factor-different-name.parquet"
    relocated.write_bytes(first.read_bytes())
    request_inputs["additional_factors"] = relocated
    assert prepared_data_request(**request_inputs) == with_factor
    relocated.write_bytes(b"second frozen factor artifact")
    assert canonical_key(prepared_data_request(**request_inputs)) != canonical_key(with_factor)


@pytest.mark.parametrize("scope", ["final_oos_opened", "inference_only", "live_retrain"])
def test_formal_or_inference_access_scope_cannot_reuse_research_cache(request_inputs, scope):
    expected = prepared_data_request(**request_inputs)
    request_inputs["manifest"][scope] = True
    if scope != "final_oos_opened":
        request_inputs["manifest"]["prediction_segment"] = "test"
    assert canonical_key(prepared_data_request(**request_inputs)) != canonical_key(expected)


def test_test_segment_uses_its_own_data_end(request_inputs):
    expected = prepared_data_request(**request_inputs)
    request_inputs["manifest"]["prediction_segment"] = "test"
    request_inputs["manifest"]["final_oos_opened"] = True
    changed = prepared_data_request(**request_inputs)
    assert expected["load_end"] == "2020-01-06"
    assert changed["load_end"] == "2020-01-07"
    assert canonical_key(changed) != canonical_key(expected)


@pytest.mark.parametrize("group,field,replacement", [
    ("manifest", "dataset_identity_sha256", "d" * 64),
    ("label_contract", "label_expression", "Ref($close,-3)/Ref($close,-1)-1"),
    ("label_contract", "horizon_sessions", 2),
    ("producer_identity", "handler_sha256", "e" * 64),
    ("producer_identity", "runtime_image_digest", "sha256:" + "f" * 64),
    ("producer_identity", "qlib_commit", "different-commit"),
    ("producer_identity", "numpy_version", "different-numpy"),
])
def test_dataset_labels_and_preparation_runtime_are_bound(
    request_inputs, group, field, replacement,
):
    expected = prepared_data_request(**request_inputs)
    request_inputs[group][field] = replacement
    assert canonical_key(prepared_data_request(**request_inputs)) != canonical_key(expected)


def test_explicit_universe_is_bound_without_implicit_market_file(request_inputs):
    request_inputs["manifest"]["universe"] = ["SH600000", "SZ000001"]
    expected = prepared_data_request(**request_inputs)
    assert expected["market_membership_sha256"] is None
    request_inputs["manifest"]["universe"] = ["SH600000"]
    assert canonical_key(prepared_data_request(**request_inputs)) != canonical_key(expected)


@pytest.mark.parametrize("invalid", ["", "not-an-identity", "../not-an-identity"])
def test_missing_or_invalid_dataset_identity_cannot_form_cache_key(request_inputs, invalid):
    request_inputs["manifest"]["dataset_identity_sha256"] = invalid
    with pytest.raises(ValueError):
        prepared_data_request(**request_inputs)


@pytest.mark.parametrize("universe", ["../cn_all", "/cn_all", "C:\\cn_all", ""])
def test_market_lookup_cannot_escape_instruments_directory(request_inputs, universe):
    request_inputs["manifest"]["universe"] = universe
    with pytest.raises(ValueError):
        prepared_data_request(**request_inputs)


@pytest.mark.parametrize("field,new_date", [
    ("train_start", "2020-01-04"), ("train_end", "2020-01-07"),
    ("train_end", "not-a-date"),
])
def test_invalid_or_reversed_train_window_is_rejected(request_inputs, field, new_date):
    request_inputs["manifest"]["periods"][field] = new_date
    with pytest.raises(ValueError):
        prepared_data_request(**request_inputs)


def test_missing_market_membership_is_rejected(request_inputs):
    market = request_inputs["provider"] / "instruments" / "cn_all.txt"
    market.unlink()
    with pytest.raises(ValueError):
        prepared_data_request(**request_inputs)


@pytest.mark.parametrize("opened", [False, None, "true", 1])
def test_sealed_test_segment_cannot_be_prepared_without_explicit_authorization(
    request_inputs, opened,
):
    request_inputs["manifest"].update(prediction_segment="test", final_oos_opened=opened)
    with pytest.raises(ValueError, match="opened final OOS ledger"):
        prepared_data_request(**request_inputs)


@pytest.mark.parametrize("mode", ["inference_only", "live_retrain"])
@pytest.mark.parametrize("invalid", ["valid_segment", "opened_oos", "multiple_days", "both_modes"])
def test_live_preparation_requires_one_non_oos_signal_date(request_inputs, mode, invalid):
    manifest = request_inputs["manifest"]
    manifest.update({mode: True, "prediction_segment": "test"})
    if invalid == "valid_segment":
        manifest["prediction_segment"] = "valid"
    elif invalid == "opened_oos":
        manifest["final_oos_opened"] = True
    elif invalid == "multiple_days":
        manifest["periods"]["test_end"] = "2020-01-08"
    else:
        manifest.update(inference_only=True, live_retrain=True)
    with pytest.raises(ValueError, match="live"):
        prepared_data_request(**request_inputs)


@pytest.mark.parametrize("invalid", ["2020-02-30", "2020-1-02", "2020-01-02T00:00:00", None])
@pytest.mark.parametrize("field", ["train_start", "valid_start", "test_end"])
def test_all_model_periods_must_be_real_iso_dates(request_inputs, field, invalid):
    request_inputs["manifest"]["periods"][field] = invalid
    with pytest.raises(ValueError, match="real ISO dates"):
        prepared_data_request(**request_inputs)


@pytest.mark.parametrize("start", ["2020-01-06", "2020-01-03"])
def test_validation_cannot_reach_or_cross_the_sealed_test_start(request_inputs, start):
    request_inputs["manifest"]["periods"]["test_start"] = start
    with pytest.raises(ValueError, match="sealed final OOS"):
        prepared_data_request(**request_inputs)


@pytest.mark.parametrize("field", ["purge_sessions", "embargo_sessions"])
@pytest.mark.parametrize("invalid", [-1, 1.0, "1", True, None])
def test_label_boundaries_must_be_nonnegative_integers(request_inputs, field, invalid):
    request_inputs["label_contract"][field] = invalid
    with pytest.raises(ValueError, match="nonnegative integers"):
        prepared_data_request(**request_inputs)


@pytest.mark.parametrize("field", ["purge_sessions", "embargo_sessions"])
def test_missing_label_boundary_is_not_silently_zero(request_inputs, field):
    request_inputs["label_contract"].pop(field)
    with pytest.raises(ValueError, match="nonnegative integers"):
        prepared_data_request(**request_inputs)


@pytest.fixture
def purged_inputs(request_inputs):
    request_inputs["manifest"]["periods"].update(
        valid_start="2020-01-07", valid_end="2020-01-08",
        test_start="2020-01-10", test_end="2020-01-10",
    )
    request_inputs["label_contract"].update(purge_sessions=1, embargo_sessions=1)
    return request_inputs


def test_positive_purge_and_embargo_allow_correctly_separated_windows(purged_inputs):
    assert prepared_data_request(**purged_inputs)["load_end"] == "2020-01-08"


@pytest.mark.parametrize("field,new_date,error", [
    ("valid_start", "2020-01-06", "train/validation boundary"),
    ("valid_end", "2020-01-07", "shorter than its label purge"),
    ("test_start", "2020-01-09", "embargo is too short"),
])
def test_each_label_boundary_is_checked_before_preparation(purged_inputs, field, new_date, error):
    purged_inputs["manifest"]["periods"][field] = new_date
    with pytest.raises(ValueError, match=error):
        prepared_data_request(**purged_inputs)


@pytest.mark.parametrize("field", ["train_end", "valid_start", "valid_end"])
def test_training_and_validation_boundaries_must_exist_in_physical_calendar(
    purged_inputs, field,
):
    calendar = purged_inputs["provider"] / "calendars" / "day.txt"
    missing = purged_inputs["manifest"]["periods"][field]
    calendar.write_text(
        "\n".join(day for day in calendar.read_text().splitlines() if day != missing) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="outside the governed trading calendar"):
        prepared_data_request(**purged_inputs)


def test_pre_final_view_does_not_require_or_invent_future_calendar(purged_inputs):
    calendar = purged_inputs["provider"] / "calendars" / "day.txt"
    calendar.write_text(
        "\n".join(day for day in calendar.read_text().splitlines() if day <= "2020-01-08")
        + "\n", encoding="utf-8",
    )
    request = prepared_data_request(**purged_inputs)
    assert request["load_end"] == "2020-01-08"
    assert "2020-01-10" not in calendar.read_text()


@pytest.mark.parametrize("field,new_date", [
    ("valid_start", "2020-01-07"), ("test_start", "2020-01-09"),
])
def test_segment_only_boundaries_do_not_fragment_prepared_array_identity(
    request_inputs, field, new_date,
):
    request_inputs["manifest"]["periods"].update(
        valid_end="2020-01-08", test_start="2020-01-10", test_end="2020-01-10",
    )
    expected = prepared_data_request(**request_inputs)
    request_inputs["manifest"]["periods"][field] = new_date
    actual = prepared_data_request(**request_inputs)
    assert actual == expected
    assert canonical_key(actual) == canonical_key(expected)
