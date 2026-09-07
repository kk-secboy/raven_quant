from __future__ import annotations

import struct

import pytest

import quant_platform.model_dataset_view_store as views
from quant_platform.model_cell_execution import execute_stored_cell, model_cell_store_root
from quant_platform.model_cell_store import ModelCellStore
from quant_platform.model_research_governance import file_sha256

pytestmark = pytest.mark.no_database


def _source(tmp_path):
    source = tmp_path / "source"
    (source / "calendars").mkdir(parents=True)
    (source / "instruments").mkdir()
    (source / "features/sh600000").mkdir(parents=True)
    (source / "calendars/day.txt").write_text("2024-01-02\n2024-01-03\n2024-01-04\n")
    (source / "instruments/cn_all.txt").write_text("SH600000\t2024-01-02\t2024-01-04\n")
    (source / "features/sh600000/close.day.bin").write_bytes(
        struct.pack("<4f", 0.0, 10.0, 11.0, 12.0)
    )
    return source


def test_cross_attempt_reuses_verified_view_and_committed_cell_without_recompute(
    tmp_path, monkeypatch,
):
    source = _source(tmp_path)
    job_root = tmp_path / "artifacts/model-evaluations/run/job"
    outputs = [job_root / "attempts" / f"attempt-000{i}-{str(i) * 32}" / "result.json"
               for i in (1, 2)]
    root = model_cell_store_root(outputs[0]).parent / "model-dataset-views"
    provider = views.prepare_model_dataset_view(source, root, cutoff="2024-01-03",
                                               provenance={"dataset_identity": "frozen"})
    code = tmp_path / "model.py"
    code.write_text("fixed candidate source")
    call = {
        "code_path": code, "provider_path": provider, "runner_path": code,
        "timeout_seconds": 7200,
        "manifest": {
            "candidate_id": "one", "seed": 11, "code_sha256": file_sha256(code),
            "model_dataset_view_receipt_sha256": file_sha256(provider.parent / "receipt.json"),
        },
    }
    fits = []

    def fit(**kwargs):
        fits.append(1)
        output = kwargs["workspace"] / "output"
        output.mkdir(parents=True)
        (output / "checkpoint.bin").write_bytes(b"fitted checkpoint")
        return {"metrics": {"ic": 0.1}}, {"source": "independently validated"}

    first = execute_stored_cell(
        ModelCellStore(model_cell_store_root(outputs[0]), {"run": "frozen"}), call,
        active_path=outputs[0].parent / "active.json", execute=fit, cleanup=lambda _: None,
    )
    monkeypatch.setattr(views.view_runtime, "_copy_truncated_day_feature", lambda *_: (
        pytest.fail("resume must not regenerate the physical bin view")
    ))
    reused_view = views.prepare_model_dataset_view(source, root, cutoff="2024-01-03",
                                                   provenance={"dataset_identity": "frozen"})
    assert reused_view == provider
    second = execute_stored_cell(
        ModelCellStore(model_cell_store_root(outputs[1]), {"run": "frozen"}), call,
        active_path=outputs[1].parent / "active.json", execute=fit, cleanup=lambda _: None,
    )
    assert second["reused"] and len(fits) == 1
    assert first["workspace"] == second["workspace"]
    assert first["receipt_sha256"] == second["receipt_sha256"]


@pytest.mark.parametrize("mutation", ["bytes", "missing", "extra"])
def test_actual_provider_bin_mutation_fails_before_cell_reuse(tmp_path, mutation):
    source = _source(tmp_path)
    root = tmp_path / "views"
    provider = views.prepare_model_dataset_view(source, root, cutoff="2024-01-03",
                                               provenance={"dataset_identity": "frozen"})
    feature = provider / "features/sh600000/close.day.bin"
    if mutation == "bytes":
        feature.write_bytes(b"changed physical feature bytes")
    elif mutation == "missing":
        feature.unlink()
    else:
        (feature.parent / "unexpected.day.bin").write_bytes(b"unexpected bytes")
    with pytest.raises(ValueError, match="verification|physical feature"):
        views.prepare_model_dataset_view(source, root, cutoff="2024-01-03",
                                         provenance={"dataset_identity": "frozen"})
