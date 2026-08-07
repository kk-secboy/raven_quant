import inspect
from datetime import date
from types import SimpleNamespace

import pytest

from quant_data.cli import (
    _exclude_superseded_specs,
    _reconcile_range_plan,
    _supersede_unplanned_range_units,
    _supersede_unsupported_governance_units,
    ashare_5m,
    bootstrap,
    snapshot,
)
from quant_data.coverage_data import (
    COVERAGE_BUNDLES,
    DEFAULT_COVERAGE_BUNDLES,
    OPTIONAL_COVERAGE_BUNDLES,
    coverage_bundle_datasets,
    coverage_primary_key_candidates,
    coverage_secondary_specs,
    coverage_specs,
)
from quant_data.models import FetchSpec, ProviderResult
from quant_data.supplemental_data import (
    bundle_datasets,
    next_pagination_specs,
    supplemental_specs,
    validate_supplemental,
)

pytestmark = pytest.mark.no_database


class _CheckpointStub:
    def __init__(
        self,
        rows: dict[str, list[dict]],
        *,
        superseded_keys: set[str] | None = None,
    ) -> None:
        self.rows = rows
        self.superseded: list[str] = []
        self.superseded_keys = superseded_keys or set()

    def successful(self, dataset: str) -> list[dict]:
        return []

    def unfinished_units(self, dataset: str) -> list[dict]:
        return list(self.rows.get(dataset, []))

    def supersede_units(self, unit_keys, reason: str) -> int:
        keys = list(unit_keys)
        self.superseded.extend(keys)
        return len(keys)

    def superseded_unit_keys(self, unit_keys) -> set[str]:
        return set(unit_keys) & self.superseded_keys


class _CountingRangeSpec:
    comparisons = 0

    def __init__(self, index: int) -> None:
        self.dataset = "cyq_chips"
        self.api_name = "cyq_chips"
        self.scope = {}
        self.params = {"ts_code": f"{index:06d}.SZ"}
        self.unit_key = str(index)

    def __eq__(self, other: object) -> bool:
        type(self).comparisons += 1
        return self is other


def test_range_plan_classifies_specs_without_quadratic_equality_scans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = [_CountingRangeSpec(index) for index in range(100)]
    checkpoint = _CheckpointStub({})
    _CountingRangeSpec.comparisons = 0
    monkeypatch.setattr("quant_data.cli.is_adaptive_partition", lambda spec: True)
    monkeypatch.setattr(
        "quant_data.cli.partition_bounds",
        lambda spec: ("date", date(2024, 1, 1), date(2024, 1, 31)),
    )

    reconciled = _reconcile_range_plan(  # type: ignore[arg-type]
        SimpleNamespace(checkpoint=checkpoint), specs
    )

    assert [spec.unit_key for spec in reconciled] == [spec.unit_key for spec in specs]
    assert _CountingRangeSpec.comparisons == 0


def test_scoped_quality_gate_is_applied_only_to_ashare_5m_command() -> None:
    assert "dataset_filter=" in inspect.getsource(ashare_5m)
    assert "dataset_filter=" not in inspect.getsource(bootstrap)
    assert "dataset_filter=" not in inspect.getsource(snapshot)


def test_coverage_inventory_matches_audited_default_and_optional_counts() -> None:
    default = set().union(
        *(coverage_bundle_datasets(bundle) for bundle in DEFAULT_COVERAGE_BUNDLES)
    )
    optional = set().union(
        *(coverage_bundle_datasets(bundle) for bundle in OPTIONAL_COVERAGE_BUNDLES)
    )
    assert len(default) == 57
    assert len(optional) == 26
    assert default.isdisjoint(optional)
    assert all(coverage_primary_key_candidates(dataset) for dataset in default | optional)


@pytest.mark.parametrize(
    ("dataset", "expected"),
    (
        ("broker_recommend", ("month", "broker", "ts_code")),
        ("ccass_hold_detail", ("ts_code", "trade_date", "col_participant_id")),
        (
            "dc_hot",
            ("trade_date", "data_type", "rank_time", "rank", "ts_code", "ts_name"),
        ),
        ("daily_info", ("trade_date", "ts_code")),
        (
            "eco_cal",
            (
                "date",
                "time",
                "currency",
                "country",
                "event",
                "value",
                "pre_value",
                "fore_value",
            ),
        ),
        ("hm_detail", ("trade_date", "ts_code", "hm_name", "hm_orgs")),
        ("idx_anns", ("url",)),
        ("moneyflow_ind_dc", ("trade_date", "content_type", "ts_code")),
        ("slb_len", ("trade_date",)),
        ("us_adjfactor", ("trade_date", "exchange", "ts_code")),
    ),
)
def test_coverage_primary_keys_match_provider_row_identity(
    dataset: str, expected: tuple[str, ...]
) -> None:
    assert coverage_primary_key_candidates(dataset)[0] == expected


@pytest.mark.parametrize("bundle", sorted(COVERAGE_BUNDLES))
def test_every_coverage_bundle_has_a_stable_task_inventory(bundle: str) -> None:
    assert bundle_datasets(bundle) == coverage_bundle_datasets(bundle)


@pytest.mark.parametrize("bundle", sorted(COVERAGE_BUNDLES - {"strategy_specialty_minutes"}))
def test_every_primary_coverage_request_is_resumable_and_truncation_safe(
    bundle: str,
) -> None:
    specs = supplemental_specs(
        bundle,
        start=date(2024, 1, 2),
        end=date(2024, 1, 2),
        trading_dates=["20240102"],
        max_attempts=5,
    )
    assert specs
    for spec in specs:
        assert spec.allow_empty is True
        assert spec.max_attempts == 5
        assert coverage_primary_key_candidates(spec.dataset)
        assert spec.scope["page_group"]
        assert int(spec.scope["page_size"]) > 0
        assert int(spec.scope["max_pages"]) > 1
        assert int(spec.scope["offset"]) == 0
        assert spec.params["limit"] == spec.scope["page_size"]
        assert spec.params["offset"] == 0


def test_default_rules_plan_full_market_cross_sections_without_stock_loops() -> None:
    for bundle in sorted(DEFAULT_COVERAGE_BUNDLES):
        specs = supplemental_specs(
            bundle,
            start=date(2024, 1, 2),
            end=date(2024, 1, 2),
            trading_dates=["20240102"],
            max_attempts=3,
        )
        datasets = {spec.dataset for spec in specs}
        expected = coverage_bundle_datasets(bundle)
        if bundle == "cn_governance_risk":
            expected = expected - {"stk_rewards", "cyq_perf"}
        if bundle == "cn_derivatives_enhanced":
            expected = expected - {"fut_index_daily"}
        assert datasets == expected
        assert all(
            "ts_code" not in spec.params for spec in specs if spec.dataset != "stock_company"
        )


def test_coverage_pagination_uses_rule_owned_page_ceiling() -> None:
    specs = supplemental_specs(
        "cn_capital_flow",
        start=date(2024, 1, 2),
        end=date(2024, 1, 2),
        trading_dates=["20240102"],
        max_attempts=3,
    )
    target = next(spec for spec in specs if spec.dataset == "moneyflow_dc")
    following = next_pagination_specs(
        specs,
        [
            {
                "unit_key": spec.unit_key,
                "row_count": spec.scope["page_size"] if spec == target else 0,
            }
            for spec in specs
        ],
    )
    page = next(spec for spec in following if spec.dataset == "moneyflow_dc")
    assert page.params["offset"] == 6_000
    assert page.scope["max_pages"] == 4


def test_derivative_index_history_is_planned_per_nh_symbol() -> None:
    specs = coverage_secondary_specs(
        "cn_derivatives_enhanced",
        {"fut_index_daily": {"NHCI.NH": (date(2024, 1, 2), date(2025, 2, 2))}},
        start=date(2024, 1, 2),
        end=date(2025, 2, 2),
        max_attempts=3,
    )

    assert len(specs) == 2
    assert {spec.api_name for spec in specs} == {"index_daily"}
    assert {spec.params["ts_code"] for spec in specs} == {"NHCI.NH"}
    assert all(spec.scope["expected_date_field"] == "trade_date" for spec in specs)


def test_bc_otc_quote_fields_are_normalized_before_date_validation() -> None:
    spec = FetchSpec(
        dataset="bc_otcqt",
        api_name="bc_otcqt",
        params={"trade_date": "20260603"},
        scope={
            "expected_date_field": "trade_date",
            "expected_date": "20260603",
        },
    )
    result = validate_supplemental(
        spec,
        ProviderResult(
            api_name="bc_otcqt",
            columns=["TRADE_DATE", "TS_CODE", "BUY_PRICE"],
            rows=[
                {
                    "TRADE_DATE": "20260603",
                    "TS_CODE": "160017.BC",
                    "BUY_PRICE": 101.25,
                }
            ],
            raw_body=b"{}",
        ),
    )

    assert result.columns == ["trade_date", "ts_code", "buy_price"]
    assert result.rows[0]["trade_date"] == "20260603"


def test_capital_flow_uses_year_month_and_daily_grains_by_density() -> None:
    dates = ["20240102", "20241231", "20250102"]
    specs = supplemental_specs(
        "cn_capital_flow",
        start=date(2024, 1, 2),
        end=date(2025, 2, 2),
        trading_dates=dates,
        max_attempts=3,
    )

    assert len([spec for spec in specs if spec.dataset == "moneyflow_hsgt"]) == 2
    assert len([spec for spec in specs if spec.dataset == "moneyflow_mkt_dc"]) == 2
    for dataset in ("moneyflow_cnt_ths", "moneyflow_ind_ths", "moneyflow_ind_dc"):
        monthly = [spec for spec in specs if spec.dataset == dataset]
        assert len(monthly) == 14
        assert all(spec.scope["partition_axis"] == "date" for spec in monthly)
    for dataset in ("moneyflow_ths", "moneyflow_dc"):
        daily = [spec for spec in specs if spec.dataset == dataset]
        assert len(daily) == len(dates)
        assert all("trade_date" in spec.params for spec in daily)


def test_only_provider_mandated_symbol_paths_expand_by_symbol() -> None:
    governance = coverage_secondary_specs(
        "cn_governance_risk",
        {
            "stk_rewards": ["000001.SZ", "600000.SH"],
            "cyq_perf": ["000001.SZ", "600000.SH"],
            "cyq_chips": ["000001.SZ", "600000.SH"],
        },
        start=date(2024, 1, 1),
        end=date(2024, 1, 31),
        max_attempts=3,
    )
    rewards = [spec for spec in governance if spec.dataset == "stk_rewards"]
    cyq = [spec for spec in governance if spec.dataset.startswith("cyq_")]
    assert len(rewards) == 1
    assert rewards[0].params["ts_code"] == "000001.SZ,600000.SH"
    assert rewards[0].scope["row_limit"] == 10_000
    assert len(cyq) == 2
    assert {spec.dataset for spec in cyq} == {"cyq_perf"}
    assert {spec.params["ts_code"] for spec in cyq} == {
        "000001.SZ",
        "600000.SH",
    }
    assert all(spec.params["start_date"] >= "20180101" for spec in cyq)
    assert all(spec.scope["partition_axis"] == "date" for spec in cyq)
    assert all(spec.scope["page_size"] == 6_000 for spec in cyq)
    assert all(spec.allow_empty is True for spec in governance)
    assert all(spec.max_attempts == 3 for spec in governance)

    minutes = coverage_secondary_specs(
        "strategy_specialty_minutes",
        {"sw_mins": ["801010.SI"], "hk_mins": ["00700.HK"]},
        start=date(2024, 1, 1),
        end=date(2024, 2, 2),
        max_attempts=3,
    )
    assert len(minutes) == 4
    assert {spec.params["freq"] for spec in minutes} == {"5min"}
    assert {spec.dataset for spec in minutes} == {"sw_mins", "hk_mins"}
    assert all(spec.allow_empty is True for spec in minutes)
    assert all(spec.max_attempts == 3 for spec in minutes)
    assert all(int(spec.scope["row_limit"]) > 0 for spec in minutes)


def test_governance_planning_respects_provider_history_and_required_symbols() -> None:
    primary = coverage_specs(
        "cn_governance_risk",
        start=date(2008, 1, 1),
        end=date(2018, 1, 2),
        trading_dates=["20080102", "20150105", "20160104", "20180102"],
        max_attempts=3,
    )
    assert not {"cyq_perf", "cyq_chips"} & {spec.dataset for spec in primary}
    ccass = [spec for spec in primary if spec.dataset.startswith("ccass_hold")]
    assert ccass
    assert all(spec.params["trade_date"] >= "20160101" for spec in ccass)

    secondary = coverage_secondary_specs(
        "cn_governance_risk",
        {
            "stk_rewards": ["000001.SZ"],
            "cyq_perf": {"000001.SZ": (date(2008, 1, 1), date(2018, 1, 2))},
            "cyq_chips": {"000001.SZ": (date(2008, 1, 1), date(2018, 1, 2))},
        },
        start=date(2008, 1, 1),
        end=date(2018, 1, 2),
        max_attempts=3,
    )
    cyq = [spec for spec in secondary if spec.dataset.startswith("cyq_")]
    assert len(cyq) == 1
    assert {spec.dataset for spec in cyq} == {"cyq_perf"}
    assert all(spec.params["ts_code"] == "000001.SZ" for spec in cyq)
    assert all(spec.params["start_date"] == "20180101" for spec in cyq)


def test_governance_cleanup_supersedes_retired_cyq_chips_units() -> None:
    legacy_spec = FetchSpec(
        dataset="cyq_chips",
        api_name="cyq_chips",
        scope={"trade_date": "20240102"},
        params={"trade_date": "20240102", "limit": 6_000, "offset": 0},
        allow_empty=True,
        max_attempts=3,
    )
    legacy = {
        "unit_key": legacy_spec.unit_key,
        "dataset": legacy_spec.dataset,
        "api_name": legacy_spec.api_name,
        "scope_json": legacy_spec.scope,
        "params_json": legacy_spec.params,
        "allow_empty": True,
        "max_attempts": 3,
    }
    checkpoint = _CheckpointStub({"cyq_chips": [legacy]})

    count = _supersede_unsupported_governance_units(SimpleNamespace(checkpoint=checkpoint))

    assert count == 1
    assert checkpoint.superseded == [legacy_spec.unit_key]


def test_range_plan_supersedes_unfinished_adaptive_children_not_in_current_plan() -> None:
    current, stale_child = [
        next(
            spec
            for spec in coverage_secondary_specs(
                "cn_governance_risk",
                {"cyq_perf": [symbol]},
                start=date(2024, 1, 1),
                end=date(2024, 12, 31),
                max_attempts=3,
            )
            if spec.dataset == "cyq_perf"
        )
        for symbol in ("000001.SZ", "000002.SZ")
    ]
    stale = {
        "unit_key": stale_child.unit_key,
        "dataset": stale_child.dataset,
        "api_name": stale_child.api_name,
        "scope_json": stale_child.scope,
        "params_json": stale_child.params,
        "allow_empty": True,
        "max_attempts": 3,
    }
    checkpoint = _CheckpointStub({"cyq_perf": [stale]})

    reconciled = _reconcile_range_plan(SimpleNamespace(checkpoint=checkpoint), [current])

    assert reconciled == [current]
    assert checkpoint.superseded == [stale_child.unit_key]


def test_executable_range_plan_guard_retires_only_absent_units() -> None:
    current, stale_child = [
        next(
            spec
            for spec in coverage_secondary_specs(
                "cn_governance_risk",
                {"cyq_perf": [symbol]},
                start=date(2024, 1, 1),
                end=date(2024, 12, 31),
                max_attempts=3,
            )
            if spec.dataset == "cyq_perf"
        )
        for symbol in ("000001.SZ", "000002.SZ")
    ]
    rows = {
        "cyq_perf": [
            {"unit_key": current.unit_key},
            {"unit_key": stale_child.unit_key},
        ]
    }
    checkpoint = _CheckpointStub(rows)

    superseded = _supersede_unplanned_range_units(
        SimpleNamespace(checkpoint=checkpoint), [current]
    )

    assert superseded == 1
    assert checkpoint.superseded == [stale_child.unit_key]


def test_pagination_completeness_excludes_superseded_audit_rows() -> None:
    current, obsolete = [
        next(
            spec
            for spec in coverage_secondary_specs(
                "cn_governance_risk",
                {"cyq_perf": [symbol]},
                start=date(2024, 1, 1),
                end=date(2024, 12, 31),
                max_attempts=3,
            )
            if spec.dataset == "cyq_perf"
        )
        for symbol in ("000001.SZ", "000002.SZ")
    ]
    checkpoint = _CheckpointStub({}, superseded_keys={obsolete.unit_key})

    executable = _exclude_superseded_specs(
        SimpleNamespace(checkpoint=checkpoint), [current, obsolete]
    )

    assert executable == [current]


def test_cyq_chips_is_not_planned_as_required_governance_data() -> None:
    specs = coverage_secondary_specs(
        "cn_governance_risk",
        {"cyq_chips": ["000001.SZ"], "cyq_perf": ["000001.SZ"]},
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
        max_attempts=3,
    )

    assert {spec.dataset for spec in specs} == {"cyq_perf"}
    assert len(specs) == 1


def test_governance_supersedes_only_pre_2016_ccass_units() -> None:
    checkpoint = _CheckpointStub(
        {
            "ccass_hold": [
                {
                    "unit_key": "unsupported",
                    "params_json": {"trade_date": "20151231"},
                },
                {
                    "unit_key": "supported",
                    "params_json": {"trade_date": "20160104"},
                },
            ]
        }
    )

    count = _supersede_unsupported_governance_units(SimpleNamespace(checkpoint=checkpoint))

    assert count == 1
    assert checkpoint.superseded == ["unsupported"]
