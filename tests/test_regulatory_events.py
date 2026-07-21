import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from quant_data.availability import (
    AvailabilityPolicyError,
    availability_policy,
    filter_available,
    recoverability_level,
)
from quant_data.qlib_builder import QlibBuilder
from quant_data.regulatory_events import (
    REGULATORY_EVENTS_RULE_VERSION,
    RegulatoryEventsError,
    classify_title,
    derive_regulatory_events,
    open_days_from_trade_cal,
)

pytestmark = pytest.mark.no_database

OPEN_DAYS = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("关于收到中国证监会《立案告知书》的公告", "csrc_investigation"),
        ("关于公司被立案调查暨风险提示的公告", "csrc_investigation"),
        ("关于收到《行政处罚决定书》的公告", "administrative_penalty"),
        ("关于收到行政处罚及市场禁入事先告知书的公告", "administrative_penalty"),
        ("关于对公司及相关责任人予以公开谴责的公告", "public_censure"),
        ("2023年年度报告", None),
        ("关于收到深圳证券交易所问询函的公告", None),
        ("关于公司及相关人员收到警示函的公告", None),
        ("关于深圳证券交易所关注函的回复公告", None),
        ("关于收到监管函的公告", None),
        ("关于公司受到通报批评纪律处分的公告", None),
        ("2023年度利润分配预案公告", None),
        ("关于筹划重大资产重组的停牌公告", None),
        (None, None),
    ],
)
def test_classify_title_conservative_rules(title: str | None, expected: str | None) -> None:
    assert classify_title(title) == expected


def test_derived_events_use_first_trading_day_after_announcement() -> None:
    announcements = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "ann_date": "20240103",
                "title": "关于收到中国证监会《立案告知书》的公告",
                "url": "http://example/1.pdf",
            }
        ]
    )
    events = derive_regulatory_events(announcements, OPEN_DAYS)
    assert len(events) == 1
    row = events.iloc[0]
    assert row["event_date"] == pd.Timestamp("2024-01-03")
    assert row["known_date"] == pd.Timestamp("2024-01-04")
    assert row["known_date"] > row["event_date"]
    assert bool(row["major"])
    assert row["event_type"] == "csrc_investigation"
    assert row["rule_version"] == REGULATORY_EVENTS_RULE_VERSION


def test_announcement_on_non_trading_day_uses_next_open_day() -> None:
    announcements = pd.DataFrame(
        [
            {
                "ts_code": "600000.SH",
                "ann_date": "2024-01-06",  # Saturday
                "title": "关于收到《行政处罚决定书》的公告",
                "url": "",
            }
        ]
    )
    open_days = [date(2024, 1, 5), date(2024, 1, 8)]
    events = derive_regulatory_events(announcements, open_days)
    assert events.iloc[0]["known_date"] == pd.Timestamp("2024-01-08")


def test_dedup_keeps_earliest_announcement_per_instrument_and_type() -> None:
    announcements = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "ann_date": "20240104",
                "title": "关于立案调查事项进展的公告",
                "url": "http://example/2.pdf",
            },
            {
                "ts_code": "000001.SZ",
                "ann_date": "20240102",
                "title": "关于收到《立案告知书》的公告",
                "url": "http://example/1.pdf",
            },
            {
                "ts_code": "000001.SZ",
                "ann_date": "20240103",
                "title": "关于收到《行政处罚决定书》的公告",
                "url": "http://example/3.pdf",
            },
        ]
    )
    events = derive_regulatory_events(announcements, OPEN_DAYS)
    assert len(events) == 2
    investigation = events[events["event_type"] == "csrc_investigation"].iloc[0]
    assert investigation["event_date"] == pd.Timestamp("2024-01-02")
    assert investigation["known_date"] == pd.Timestamp("2024-01-03")
    penalty = events[events["event_type"] == "administrative_penalty"].iloc[0]
    assert penalty["known_date"] == pd.Timestamp("2024-01-04")


def test_clean_announcements_yield_empty_events_with_schema() -> None:
    announcements = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "ann_date": "20240102", "title": "2023年年度报告", "url": "u"},
        ]
    )
    events = derive_regulatory_events(announcements, OPEN_DAYS)
    assert events.empty
    assert {"ts_code", "event_date", "known_date", "major"}.issubset(events.columns)


def test_announcement_beyond_calendar_fails_closed() -> None:
    announcements = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "ann_date": "20240105",
                "title": "关于收到《立案告知书》的公告",
                "url": "u",
            }
        ]
    )
    with pytest.raises(RegulatoryEventsError):
        derive_regulatory_events(announcements, OPEN_DAYS)


def test_missing_columns_fail_closed() -> None:
    with pytest.raises(RegulatoryEventsError):
        derive_regulatory_events(pd.DataFrame([{"ts_code": "000001.SZ"}]), OPEN_DAYS)


def test_open_days_from_trade_cal_requires_open_days() -> None:
    frame = pd.DataFrame(
        [
            {"cal_date": "20240102", "is_open": "1"},
            {"cal_date": "20240106", "is_open": "0"},
        ]
    )
    assert open_days_from_trade_cal(frame) == [date(2024, 1, 2)]
    with pytest.raises(RegulatoryEventsError):
        open_days_from_trade_cal(pd.DataFrame([{"cal_date": "20240106", "is_open": "0"}]))
    with pytest.raises(RegulatoryEventsError):
        open_days_from_trade_cal(pd.DataFrame([{"cal_date": "20240102"}]))


def test_availability_registry_guards_regulatory_events() -> None:
    policy = availability_policy("regulatory_events")
    assert policy is not None and policy.date_columns == ("known_date",)
    assert recoverability_level("regulatory_events") == "reconstructed"
    events = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "event_date": "2024-01-03",
                "known_date": "2024-01-04",
                "major": True,
            }
        ]
    )
    assert filter_available("regulatory_events", events, "2024-01-03").empty
    assert len(filter_available("regulatory_events", events, "2024-01-04")) == 1
    with pytest.raises(AvailabilityPolicyError):
        filter_available("regulatory_events", events.drop(columns=["known_date"]), "2024-01-04")


def _write_snapshot_frame(snapshot: Path, dataset: str, frame: pd.DataFrame) -> None:
    target = snapshot / "parquet" / dataset
    target.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target / "data.parquet", index=False)


def _eligibility_snapshot(tmp_path: Path, *, with_anns_d: bool) -> Path:
    """Snapshot where SZ000001 is fully eligible except for a violation event."""

    snapshot = tmp_path / "snapshot"
    trading_days = pd.bdate_range("2023-10-09", "2024-01-10")
    daily = pd.DataFrame(
        {
            "ts_code": "000001.SZ",
            "trade_date": trading_days,
            "amount": 600_000.0,
            "vol": 100.0,
        }
    )
    _write_snapshot_frame(snapshot, "daily", daily)
    _write_snapshot_frame(
        snapshot,
        "stock_basic",
        pd.DataFrame(
            [{"ts_code": "000001.SZ", "list_date": "2020-01-02", "delist_date": None}]
        ),
    )
    _write_snapshot_frame(
        snapshot,
        "balancesheet",
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "ann_date": "2023-10-09",
                    "total_hldr_eqy_exc_min_int": 10_000_000.0,
                }
            ]
        ),
    )
    _write_snapshot_frame(
        snapshot,
        "fina_audit",
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "ann_date": "2023-10-09",
                    "audit_result": "standard_unqualified",
                }
            ]
        ),
    )
    calendar_days = trading_days.union(pd.bdate_range("2024-01-11", "2024-01-12"))
    _write_snapshot_frame(
        snapshot,
        "trade_cal",
        pd.DataFrame({"cal_date": calendar_days, "is_open": 1}),
    )
    if with_anns_d:
        _write_snapshot_frame(
            snapshot,
            "anns_d",
            pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "ann_date": "20240103",
                        "title": "关于收到中国证监会《立案告知书》的公告",
                        "url": "http://example/violation.pdf",
                    },
                    {
                        "ts_code": "000001.SZ",
                        "ann_date": "20231220",
                        "title": "2023年第三季度报告",
                        "url": "http://example/report.pdf",
                    },
                ]
            ),
        )
    return snapshot


def test_qlib_build_excludes_violator_strictly_after_announcement(tmp_path: Path) -> None:
    snapshot = _eligibility_snapshot(tmp_path, with_anns_d=True)
    target = tmp_path / "qlib"

    assert QlibBuilder(snapshot)._write_eligibility_metadata(target)

    matrix = pd.read_parquet(target / "eligibility_matrix.parquet")
    assert bool(matrix["regulatory_data_available"].all())
    row = matrix.set_index("datetime")
    before = row.loc["2024-01-03"]
    after = row.loc["2024-01-04"]
    assert not bool(before["major_violation"])
    assert bool(before["eligible"])
    assert bool(after["major_violation"])
    assert not bool(after["eligible"])
    assert "major_violation" in json.loads(after["reasons"])
    # The exclusion is permanent: still flagged on the last observed date.
    last = row.iloc[-1]
    assert bool(last["major_violation"])
    contract = json.loads((target / "eligibility_contract.json").read_text(encoding="utf-8"))
    assert contract["regulatory_data_available"] is True
    assert contract["regulatory_origin"] == (
        f"anns_d_title_rules({REGULATORY_EVENTS_RULE_VERSION})"
    )


def test_qlib_build_with_anns_d_but_no_violations_marks_data_available(tmp_path: Path) -> None:
    snapshot = _eligibility_snapshot(tmp_path, with_anns_d=False)
    _write_snapshot_frame(
        snapshot,
        "anns_d",
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "ann_date": "20231220",
                    "title": "2023年第三季度报告",
                    "url": "http://example/report.pdf",
                }
            ]
        ),
    )
    target = tmp_path / "qlib"

    assert QlibBuilder(snapshot)._write_eligibility_metadata(target)

    matrix = pd.read_parquet(target / "eligibility_matrix.parquet")
    # Rules ran over the full anns_d index and found nothing: the regulatory
    # feed is considered available with zero exclusions.
    assert bool(matrix["regulatory_data_available"].all())
    assert not bool(matrix["major_violation"].any())


def test_qlib_build_without_anns_d_keeps_fail_soft_behavior(tmp_path: Path) -> None:
    snapshot = _eligibility_snapshot(tmp_path, with_anns_d=False)
    target = tmp_path / "qlib"

    assert QlibBuilder(snapshot)._write_eligibility_metadata(target)

    matrix = pd.read_parquet(target / "eligibility_matrix.parquet")
    assert not bool(matrix["regulatory_data_available"].any())
    assert not bool(matrix["major_violation"].any())
    contract = json.loads((target / "eligibility_contract.json").read_text(encoding="utf-8"))
    assert contract["regulatory_data_available"] is False
    assert contract["regulatory_origin"] is None


def test_materialized_dataset_takes_precedence_over_derivation(tmp_path: Path) -> None:
    snapshot = _eligibility_snapshot(tmp_path, with_anns_d=True)
    _write_snapshot_frame(
        snapshot,
        "regulatory_events",
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "event_date": "2023-12-20",
                    "known_date": "2023-12-21",
                    "major": True,
                }
            ]
        ),
    )
    target = tmp_path / "qlib"

    assert QlibBuilder(snapshot)._write_eligibility_metadata(target)

    matrix = pd.read_parquet(target / "eligibility_matrix.parquet").set_index("datetime")
    assert bool(matrix.loc["2023-12-21", "major_violation"])
    contract = json.loads((target / "eligibility_contract.json").read_text(encoding="utf-8"))
    assert contract["regulatory_origin"] == "materialized_dataset"
