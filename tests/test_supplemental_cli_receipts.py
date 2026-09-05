"""Supplemental command receipts must include its completed secondary phases."""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest
from typer.testing import CliRunner

from quant_data import cli
from quant_data.supplemental_data import bond_reference_specs, bundle_datasets

pytestmark = pytest.mark.no_database


@pytest.mark.parametrize("reuse_references", [False, True])
def test_bond_receipt_accounts_for_all_completed_reference_units(
    tmp_path, monkeypatch, reuse_references
) -> None:
    start, end = date(2026, 8, 28), date(2026, 9, 4)
    symbols = [f"{index:06d}.SH" for index in range(1, 102)]
    references = bond_reference_specs(symbols, start=start, end=end, max_attempts=3)
    reference_datasets = {spec.dataset for spec in references}
    # Include a legitimate empty endpoint and multiple unequal-sized batches.
    row_counts = {"cb_price_chg": 2, "cb_rate": 3, "cb_rating": 0,
                  "cb_share": 5, "top10_cb_holders": 7}
    planned = {}
    completed = {}
    fetched_references = []

    def complete(spec):
        completed[spec.unit_key] = {
            "unit_key": spec.unit_key, "dataset": spec.dataset,
            "row_count": row_counts.get(spec.dataset, 11), "status": "succeeded",
        }

    if reuse_references:
        for spec in references:
            complete(spec)

    def add(specs):
        values = list(specs)
        inserted = sum(spec.unit_key not in completed for spec in values)
        planned.update({spec.unit_key: spec for spec in values})
        return inserted

    checkpoint = SimpleNamespace(
        add=add,
        retry_failed_units=lambda keys: list(keys),
        successful=lambda dataset: [row for row in completed.values() if row["dataset"] == dataset],
        successful_units=lambda keys: [completed[key] for key in keys if key in completed],
    )
    context = SimpleNamespace(
        settings=SimpleNamespace(max_request_attempts=3),
        planner=SimpleNamespace(trading_dates=lambda *_args: [
            "20260828", "20260831", "20260901", "20260902", "20260903", "20260904",
        ]),
        checkpoint=checkpoint,
        storage=SimpleNamespace(read_units=lambda _rows: pd.DataFrame({"ts_code": symbols})),
        report_progress=lambda *_args, **_kwargs: None,
    )

    def primary_phase(_context, _label, specs):
        inserted = add(specs)
        for spec in specs:
            complete(spec)
        return list(specs), checkpoint.successful_units({spec.unit_key for spec in specs}), inserted

    def reference_phase(_context, _label, datasets):
        assert datasets == reference_datasets
        for spec in planned.values():
            if spec.dataset in datasets and spec.unit_key not in completed:
                fetched_references.append(spec.unit_key)
                complete(spec)

    monkeypatch.setattr(cli, "load_context", lambda **_kwargs: context)
    monkeypatch.setattr(cli, "_run_paginated_specs", primary_phase)
    monkeypatch.setattr(cli, "_run_phase", reference_phase)
    # Keep the real _require_specs_complete, secondary planner and receipt writer.
    result_path = tmp_path / "receipt.json"
    invocation = CliRunner().invoke(cli.app, [
        "supplemental-download", "--bundle", "cn_options_bonds",
        "--start", start.isoformat(), "--end", end.isoformat(), "--result", str(result_path),
    ])
    assert invocation.exit_code == 0, invocation.output
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "succeeded"
    assert result["pagination_verified"] is True
    assert set(result["datasets"]) == bundle_datasets("cn_options_bonds")
    for dataset in reference_datasets:
        expected_units = sum(spec.dataset == dataset for spec in references)
        assert result["datasets"][dataset] == {
            "units": expected_units, "rows": expected_units * row_counts[dataset],
        }
    assert sum(value["units"] for value in result["datasets"].values()) == result["units"]
    assert sum(value["rows"] for value in result["datasets"].values()) == result["rows"]
    assert result["units"] == len(completed)
    assert result["rows"] == sum(row["row_count"] for row in completed.values())
    assert len(fetched_references) == (0 if reuse_references else len(references))
