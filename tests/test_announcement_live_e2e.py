"""Live end-to-end proof of the announcement -> NLP -> PIT mainline.

Requires OPENAI_API_KEY in the environment (skip otherwise). Runs fully against
real services with no mocks:

- discovery: Tushare anns_d/trade_cal when TUSHARE_TOKEN is configured, else
  cninfo's own public announcement query API (with a weekday-calendar fixture
  for the reference calendar, since only Tushare provides trade_cal here)
- real cninfo PDF download into the immutable store
- real LLM structuring via the announcement_nlp pipeline
- PIT assertions: fields carry available_at/ingested_at and stay invisible to a
  decision made before available_at
"""

from __future__ import annotations

import json
import os
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from quant_data.catalog import REFERENCE_FIELDS
from quant_data.cninfo_announcements import (
    download_cninfo_announcements,
    load_trade_calendar_open_days,
    next_trading_day,
)
from quant_data.provider import TushareHttpProvider
from quant_data.rate_limit import GlobalRateGate
from quant_platform.announcement_nlp import FACTOR_NAME, process_announcements

TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN") or os.environ.get("TUSHARE_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

pytestmark = [
    pytest.mark.no_database,
    pytest.mark.skipif(
        not OPENAI_API_KEY,
        reason="live e2e requires OPENAI_API_KEY in the environment",
    ),
]

CNINFO_QUERY_URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
CNINFO_STATIC = "http://static.cninfo.com.cn/"
CN_TZ = timezone(timedelta(hours=8))


def _write_unit(data_root: Path, dataset: str, frame: pd.DataFrame) -> None:
    target = data_root / "units" / dataset
    target.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target / f"{dataset}_live.parquet", index=False)


def _tushare_announcements(provider: TushareHttpProvider, days: int = 14) -> pd.DataFrame:
    today = date.today()
    for offset in range(1, days + 1):
        day = today - timedelta(days=offset)
        result = provider.fetch("anns_d", {"ann_date": day.strftime("%Y%m%d")}, ())
        frame = pd.DataFrame(result.rows, columns=result.columns)
        if not frame.empty and frame["url"].notna().any():
            return frame[frame["url"].notna()].head(3).reset_index(drop=True)
    pytest.skip("no Tushare anns_d announcements with urls found in the recent window")


def _cninfo_announcements(days: int = 14) -> pd.DataFrame:
    today = date.today()
    for offset in range(0, days):
        day = today - timedelta(days=offset)
        if day.weekday() >= 5:
            continue
        payload = (
            "pageNum=1&pageSize=5&column=szse&tabName=fulltext"
            f"&seDate={day.isoformat()}~{day.isoformat()}"
        ).encode()
        request = urllib.request.Request(
            CNINFO_QUERY_URL,
            data=payload,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.loads(response.read().decode())
        rows = []
        for item in data.get("announcements") or []:
            code = item["secCode"]
            suffix = ".SH" if code.startswith("6") else (".BJ" if code[0] in "48" else ".SZ")
            ann_day = datetime.fromtimestamp(item["announcementTime"] / 1000, CN_TZ).date()
            rows.append(
                {
                    "ts_code": f"{code}{suffix}",
                    "ann_date": ann_day,
                    "title": item["announcementTitle"],
                    "url": f"{CNINFO_STATIC}{item['adjunctUrl']}",
                }
            )
        if rows:
            return pd.DataFrame(rows[:3])
    pytest.skip("no cninfo announcements found in the recent window")


def _weekday_calendar(data_root: Path, center: date) -> None:
    """Reference calendar fixture (Mon-Fri), used only when Tushare trade_cal is absent."""
    rows = []
    day = center - timedelta(days=10)
    while day <= center + timedelta(days=10):
        rows.append(
            {
                "exchange": "SSE",
                "cal_date": day,
                "is_open": 1 if day.weekday() < 5 else 0,
                "pretrade_date": day - timedelta(days=1),
            }
        )
        day += timedelta(days=1)
    _write_unit(data_root, "trade_cal", pd.DataFrame(rows))


def test_live_announcement_nlp_pit_mainline(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    gate = GlobalRateGate(60.0)

    # 1. real discovery + trading calendar
    if TUSHARE_TOKEN:
        provider = TushareHttpProvider(
            api_url=os.environ.get("TUSHARE_API_URL", "https://api.tushare.pro"),
            token=TUSHARE_TOKEN,
            rate_gate=gate,
            timeout_seconds=60.0,
            max_attempts=3,
        )
        announcements = _tushare_announcements(provider)
        ann_day = pd.to_datetime(announcements["ann_date"].iloc[0]).date()
        calendar = provider.fetch(
            "trade_cal",
            {
                "exchange": "SSE",
                "start_date": (ann_day - timedelta(days=10)).strftime("%Y%m%d"),
                "end_date": (ann_day + timedelta(days=10)).strftime("%Y%m%d"),
            },
            REFERENCE_FIELDS["trade_cal"],
        )
        _write_unit(data_root, "trade_cal", pd.DataFrame(calendar.rows, columns=calendar.columns))
    else:
        announcements = _cninfo_announcements()
        ann_day = pd.to_datetime(announcements["ann_date"].iloc[0]).date()
        _weekday_calendar(data_root, ann_day)
    _write_unit(data_root, "anns_d", announcements)

    # 2. real cninfo download (limit=1 keeps it to a single PDF)
    download = download_cninfo_announcements(data_root, limit=1, rate_gate=gate)
    assert download.as_dict()["failed"] == 0
    index = pd.read_parquet(data_root / "announcements" / "index.parquet")
    assert len(index) == 1
    row = index.iloc[0]
    open_days = load_trade_calendar_open_days(data_root)
    assert row["available_at"] == pd.Timestamp(
        next_trading_day(pd.Timestamp(row["ann_date"]).date(), open_days)
    )
    assert row["available_at"] > pd.Timestamp(row["ann_date"])
    assert row["ingested_at"].tzinfo is not None
    assert (data_root / row["file_path"]).read_bytes()[:4] == b"%PDF"

    # 3. real LLM structuring via env-fallback credentials (one call)
    summary = process_announcements(data_root, limit=1, rate_gate=gate, environ=dict(os.environ))
    result = summary.as_dict() if hasattr(summary, "as_dict") else dict(summary)
    assert result.get("failed", 0) == 0, result

    fields = pd.read_parquet(data_root / "announcements" / "nlp" / "fields.parquet")
    assert len(fields) == 1
    field = fields.iloc[0]
    assert field["available_at"] == row["available_at"]
    assert pd.Timestamp(field["ingested_at"]).tzinfo is not None
    assert -1.0 <= float(field["tone_score"]) <= 1.0
    assert field["event_type"]

    # 4. factor artifact carries PIT timestamps and is invisible before available_at
    artifact_path = data_root / "announcements" / "nlp" / "factors" / f"{FACTOR_NAME}.parquet"
    artifact = pd.read_parquet(artifact_path)
    assert len(artifact) == 1
    assert pd.Timestamp(artifact["datetime"].iloc[0]) == pd.Timestamp(field["available_at"])
    decision_before = pd.Timestamp(field["available_at"]) - pd.offsets.Day(1)
    assert artifact[artifact["datetime"] <= decision_before].empty
    assert not artifact[artifact["datetime"] <= pd.Timestamp(field["available_at"])].empty
