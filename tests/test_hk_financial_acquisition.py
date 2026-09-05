from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_data import cli
from quant_data.availability import (
    UNAVAILABLE,
    AvailabilityPolicyError,
    filter_available,
    recoverability_level,
)
from quant_data.models import FetchSpec, ProviderResult
from quant_data.provider import ProviderError
from quant_data.supplemental_data import market_financial_specs, validate_supplemental

pytestmark = pytest.mark.no_database
DATASETS = ("hk_income", "hk_balancesheet", "hk_cashflow", "hk_fina_indicator")


def _specs(start=date(2026, 8, 28), end=date(2026, 9, 4)):
    return market_financial_specs(
        "hk", ["00700.HK"], start=start, end=end, max_attempts=3,
    )


def test_incremental_hk_refresh_does_not_exclude_older_or_nonstandard_report_periods():
    # Any of these periods can acquire a late disclosure/correction this week.
    # The provider filters accounting periods, and publishes no maximum lag.
    periods = ("19991231", "20240630", "20260531", "20260630", "20260725", "20260831")
    specs = _specs()
    assert {spec.dataset for spec in specs} == set(DATASETS)
    for spec in specs:
        assert "start_date" not in spec.params
        assert "period" not in spec.params
        assert "report_type" not in spec.params
        assert spec.params == {"ts_code": "00700.HK", "end_date": "20260904"}
        assert all(period <= spec.params["end_date"] for period in periods)
        assert spec.scope["as_of"] == "2026-09-04"
        assert spec.scope["report_period_query"] == "all-history-through-as-of-v1"


def test_hk_refresh_reuses_same_asof_but_never_legacy_narrow_window():
    initial, same_asof, next_day = _specs(), _specs(date(2008, 1, 1)), _specs(end=date(2026, 9, 5))
    for old, same, next_spec in zip(initial, same_asof, next_day, strict=True):
        legacy_params = {"ts_code": "00700.HK", "start_date": "20260828", "end_date": "20260904"}
        legacy = FetchSpec(
            old.dataset, old.api_name, {**legacy_params, "row_limit": 10_000}, legacy_params,
        )
        assert old.unit_key == same.unit_key
        assert old.unit_key != next_spec.unit_key
        assert old.unit_key != legacy.unit_key
        assert old.params["end_date"] != next_spec.params["end_date"]


@pytest.mark.parametrize("dataset", DATASETS)
def test_hk_full_history_still_rejects_possible_provider_truncation(dataset):
    spec = next(spec for spec in _specs() if spec.dataset == dataset)
    expected_limit = 200 if dataset == "hk_fina_indicator" else 10_000
    assert spec.scope["row_limit"] == expected_limit
    rows = [{"ts_code": "00700.HK", "end_date": "20260630"}] * expected_limit
    with pytest.raises(ProviderError, match="may be truncated"):
        validate_supplemental(spec, ProviderResult(spec.api_name, [], rows, b"{}"))


@pytest.mark.parametrize("dataset", DATASETS)
def test_acquisition_never_turns_report_period_into_pit_availability(dataset):
    spec = next(spec for spec in _specs() if spec.dataset == dataset)
    # No announcement evidence is synthesized by acquisition. The existing
    # unregistered availability policy must continue rejecting formal PIT use.
    response = ProviderResult(spec.api_name, ["ts_code", "end_date"], [
        {"ts_code": "00700.HK", "end_date": "20260630"},
    ], b"{}")
    assert validate_supplemental(spec, response) is response
    assert "ann_date" not in response.rows[0] and "available_at" not in response.rows[0]
    assert recoverability_level(dataset) == UNAVAILABLE
    with pytest.raises(AvailabilityPolicyError, match="no declared availability policy"):
        filter_available(dataset, pd.DataFrame(response.rows), "2026-09-05")


def test_us_indicator_refresh_covers_nonstandard_periods_and_preserves_notice_date():
    def indicator(start, end):
        return next(spec for spec in market_financial_specs(
            "us", ["NVDA"], start=start, end=end, max_attempts=3,
        ) if spec.dataset == "us_fina_indicator")

    current = indicator(date(2026, 8, 28), date(2026, 9, 4))
    same_asof = indicator(date(2008, 1, 1), date(2026, 9, 4))
    next_day = indicator(date(2026, 8, 28), date(2026, 9, 5))
    assert current.params == {"ts_code": "NVDA", "end_date": "20260904"}
    assert current.scope["as_of"] == "2026-09-04"
    assert current.unit_key == same_asof.unit_key
    assert current.unit_key != next_day.unit_key
    # The official example distinguishes a nonstandard fiscal end (July 28)
    # from its later notice (August 28). Neither field may be synthesized.
    response = ProviderResult(current.api_name, ["end_date", "notice_date"], [
        {"end_date": "20240728", "notice_date": "20240828"},
    ], b"{}")
    assert validate_supplemental(current, response) is response
    assert response.rows == [{"end_date": "20240728", "notice_date": "20240828"}]
    assert recoverability_level(current.dataset) == UNAVAILABLE
    with pytest.raises(AvailabilityPolicyError, match="no declared availability policy"):
        filter_available(current.dataset, pd.DataFrame(response.rows), "2026-09-05")
    assert current.scope["row_limit"] == 200
    with pytest.raises(ProviderError, match="may be truncated"):
        validate_supplemental(
            current, ProviderResult(current.api_name, [], response.rows * 200, b"{}"),
        )


class _Checkpoint:
    def __init__(self, rows=()):
        self.rows = {row["unit_key"]: dict(row) for row in rows}
        self.retired = []

    def add(self, specs):
        inserted = 0
        for spec in specs:
            if spec.unit_key not in self.rows:
                self.rows[spec.unit_key] = _row(spec)
                inserted += 1
        return inserted

    def unit_rows(self, keys):
        return [self.rows[key] for key in keys if key in self.rows]

    def unfinished_units(self, datasets):
        return [row for row in self.rows.values()
                if row["dataset"] in datasets and row["status"] in {"pending", "failed"}]

    def supersede_units(self, keys, _reason):
        for key in keys:
            assert self.rows[key]["status"] in {"pending", "failed"}
            self.rows[key]["status"] = "superseded"
            self.retired.append(key)
        return len(keys)

    def superseded_unit_keys(self, keys):
        return {row["unit_key"] for row in self.unit_rows(keys) if row["status"] == "superseded"}

    def successful_units(self, keys):
        return [row for row in self.unit_rows(keys) if row["status"] == "succeeded"]


def _row(spec, status="pending"):
    return {
        "unit_key": spec.unit_key, "dataset": spec.dataset, "api_name": spec.api_name,
        "params_json": spec.params, "scope_json": spec.scope, "fields_json": list(spec.fields),
        "status": status, "row_count": 0, "allow_empty": True, "max_attempts": 3,
    }


def _legacy(current, *, start="20260828", end="20260904", symbol=None, api_name=None):
    params = {"ts_code": symbol or current.params["ts_code"], "start_date": start, "end_date": end}
    spec = FetchSpec(
        current.dataset, api_name or current.api_name,
        {**params, "row_limit": 200 if current.dataset.startswith("us_") else 10_000},
        params, allow_empty=True, max_attempts=3,
    )
    return spec


@pytest.mark.parametrize("dataset", [*DATASETS, "us_fina_indicator"])
@pytest.mark.parametrize("status", ["pending", "failed"])
def test_refresh_execution_retires_old_window_only_after_durable_replacement(
    monkeypatch, dataset, status,
):
    market = "hk" if dataset.startswith("hk_") else "us"
    current = next(spec for spec in market_financial_specs(
        market, ["00700.HK" if market == "hk" else "NVDA"],
        start=date(2026, 8, 28), end=date(2026, 9, 4), max_attempts=3,
    ) if spec.dataset == dataset)
    obsolete = _legacy(current)
    successful = _row(_legacy(current, start="20240101"), "succeeded")
    successful.update(row_count=7, output_path="old.parquet", sha256="immutable")
    checkpoint = _Checkpoint([_row(obsolete, status), successful])
    context = SimpleNamespace(checkpoint=checkpoint, report_progress=lambda *a, **kw: None)
    executed = []

    def run_phase(_context, _label, datasets):
        # The real runner claims by dataset, so stale keys must be retired
        # before it starts; filtering just the returned specs is insufficient.
        for row in checkpoint.unfinished_units(datasets):
            executed.append(row["unit_key"])
            row.update(status="succeeded", row_count=2)

    monkeypatch.setattr(cli, "_run_phase", run_phase)
    specs, rows, inserted = cli._run_paginated_specs(context, "financials", [current])
    assert checkpoint.retired == [obsolete.unit_key]
    assert executed == [current.unit_key]
    assert checkpoint.rows[successful["unit_key"]] == successful
    assert specs == [current] and inserted == 1 and len(rows) == 1
    assert rows[0]["row_count"] == 2
    # The same as-of run reuses the successful full-history result.
    _specs_again, _rows_again, inserted_again = cli._run_paginated_specs(
        context, "financials", [current],
    )
    assert inserted_again == 0 and executed == [current.unit_key]


def test_window_retirement_preserves_other_symbols_asofs_and_request_shapes():
    current = _specs()[0]
    obsolete = _legacy(current)
    protected = [
        _legacy(current, symbol="00005.HK"),
        _legacy(current, end="20260903"),
        _legacy(current, end="20260905"),
        _legacy(current, api_name="other_api"),
    ]
    scoped = _legacy(current, start="20260101")
    protected.append(FetchSpec(
        scoped.dataset, scoped.api_name, scoped.scope, {**scoped.params, "report_type": "3"},
    ))
    protected.append(FetchSpec(
        obsolete.dataset, obsolete.api_name, obsolete.scope, obsolete.params,
        fields=("ts_code", "end_date", "custom_nondefault_field"),
    ))
    checkpoint = _Checkpoint([_row(obsolete), *[_row(spec) for spec in protected]])
    checkpoint.add([current])
    cli._supersede_legacy_report_period_windows(SimpleNamespace(checkpoint=checkpoint), [current])
    assert checkpoint.retired == [obsolete.unit_key]
    assert all(checkpoint.rows[spec.unit_key]["status"] == "pending" for spec in protected)


@pytest.mark.parametrize("replacement_status", [None, "superseded"])
def test_window_retirement_requires_active_durable_replacement(replacement_status):
    current = _specs()[0]
    obsolete = _legacy(current)
    checkpoint = _Checkpoint([_row(obsolete)])
    if replacement_status:
        checkpoint.rows[current.unit_key] = _row(current, replacement_status)
    with pytest.raises(RuntimeError, match="durable report-period replacement"):
        cli._supersede_legacy_report_period_windows(
            SimpleNamespace(checkpoint=checkpoint), [current],
        )
    assert checkpoint.retired == []
    assert checkpoint.rows[obsolete.unit_key]["status"] == "pending"


def test_window_retirement_never_runs_before_successful_checkpoint_add(monkeypatch):
    current = _specs()[0]
    obsolete = _legacy(current)
    checkpoint = _Checkpoint([_row(obsolete)])
    context = SimpleNamespace(checkpoint=checkpoint, report_progress=lambda *a, **kw: None)

    def crash_add(_specs):
        raise RuntimeError("checkpoint add failed")

    monkeypatch.setattr(checkpoint, "add", crash_add)
    with pytest.raises(RuntimeError, match="checkpoint add failed"):
        cli._run_paginated_specs(context, "financials", [current])
    assert checkpoint.retired == []
    assert checkpoint.rows[obsolete.unit_key]["status"] == "pending"
