"""Data-download and ingestion command builders for LocalJobWorker."""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

from ..announcement_nlp import (
    DEFAULT_BATCH_SIZE as ANNOUNCEMENT_DEFAULT_BATCH_SIZE,
)
from ..announcement_nlp import (
    DEFAULT_WORKERS as ANNOUNCEMENT_DEFAULT_WORKERS,
)
from ..corpus_nlp import (
    DEFAULT_BATCH_SIZE as CORPUS_DEFAULT_BATCH_SIZE,
)
from ..corpus_nlp import (
    DEFAULT_IRM_PER_INSTRUMENT_DAY as CORPUS_DEFAULT_IRM_PER_INSTRUMENT_DAY,
)
from ..corpus_nlp import (
    DEFAULT_MAJOR_NEWS_PER_DAY as CORPUS_DEFAULT_MAJOR_NEWS_PER_DAY,
)
from ..corpus_nlp import (
    DEFAULT_WORKERS as CORPUS_DEFAULT_WORKERS,
)


def research_asset_acquire_command(
    worker, job: dict
) -> tuple[list[str], Path | None, dict[str, str]]:
    payload = job["payload"]
    output = (
        worker.settings.data_root
        / "artifacts"
        / "research-asset-acquisitions"
        / job["id"]
    )
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    mode = str(payload.get("mode") or "")
    command = [sys.executable, "-m", "quant_data.cli"]
    if mode == "automatic":
        snapshot_name = str(payload.get("snapshot_name") or "")
        if (
            not snapshot_name
            or snapshot_name in {".", ".."}
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", snapshot_name)
        ):
            raise ValueError("automatic research asset snapshot identity is invalid")
        research_day = date.fromisoformat(str(payload.get("as_of") or ""))
        if not payload.get("include_tushare") and not payload.get("include_arxiv"):
            raise ValueError("automatic research asset job has no enabled source")
        command.extend(
            [
                "research-assets",
                "--snapshot",
                snapshot_name,
                "--as-of",
                research_day.isoformat(),
                "--result",
                str(result_path),
            ]
        )
        if payload.get("tushare_report_date"):
            report_date = date.fromisoformat(str(payload["tushare_report_date"]))
            command.extend(["--tushare-report-date", report_date.isoformat()])
        if not payload.get("include_tushare"):
            command.append("--skip-tushare")
        if not payload.get("include_arxiv"):
            command.append("--skip-arxiv")
    elif mode == "manual_https":
        document_kind = str(payload.get("document_kind") or "")
        asset_type = {
            "paper": "manual_paper",
            "research_report": "research_report",
        }.get(document_kind)
        if asset_type is None:
            raise ValueError("manual research asset document kind is invalid")
        command.extend(
            [
                "research-asset-fetch-pdf",
                "--url",
                str(payload["url"]),
                "--title",
                str(payload["title"]),
                "--type",
                asset_type,
                "--result",
                str(result_path),
            ]
        )
        if payload.get("published_at"):
            command.extend(["--published-at", str(payload["published_at"])])
    else:
        raise ValueError("research asset acquisition mode is invalid")
    return command, result_path, {}


def baostock_overlap_validation_command(
    worker, job: dict
) -> tuple[list[str], Path | None, dict[str, str]]:
    payload = job["payload"]
    result_path = Path(payload["result_path"])
    command = [
        sys.executable,
        "-m",
        "quant_data.cli",
        "validate-baostock-overlap",
        "--start",
        payload["start"],
        "--end",
        payload["end"],
        "--result",
        str(result_path),
    ]
    if payload.get("symbols"):
        command.extend(["--symbols", ",".join(payload["symbols"])])
    return command, result_path, {}


def legacy_market_backfill_command(
    worker, job: dict
) -> tuple[list[str], Path | None, dict[str, str]]:
    payload = job["payload"]
    result_path = Path(payload["result_path"])
    command = [
        sys.executable,
        "-m",
        "quant_data.cli",
        "bootstrap-legacy-market",
        "--start",
        payload["start"],
        "--end",
        payload["end"],
        "--validation-report",
        payload["validation_report"],
        "--result",
        str(result_path),
    ]
    return command, result_path, {}


def cninfo_announcements_download_command(
    worker, job: dict
) -> tuple[list[str], Path | None, dict[str, str]]:
    payload = job["payload"]
    output = worker.settings.data_root / "artifacts" / "execution-data" / job["id"]
    result_path = output / "result.json"
    command = [
        sys.executable,
        "-m",
        "quant_data.cli",
        "cninfo-announcements",
        "--start",
        str(payload["start"]),
        "--end",
        str(payload["end"]),
        "--result",
        str(result_path),
    ]
    if payload.get("ts_codes"):
        command.extend(["--ts-code", ",".join(payload["ts_codes"])])
    if int(payload.get("limit") or 0) > 0:
        command.extend(["--limit", str(payload["limit"])])
    if payload.get("regulatory_only", True):
        command.append("--regulatory-only")
    return command, result_path, {}


def nlp_download_command(
    worker, job: dict
) -> tuple[list[str], Path | None, dict[str, str]]:
    payload = job["payload"]
    output = worker.settings.data_root / "artifacts" / "execution-data" / job["id"]
    result_path = output / "result.json"
    command = [sys.executable, "-m", "quant_data.cli"]
    if job["kind"] == "announcement_nlp":
        command.extend(
            [
                "announcement-nlp",
                "--start",
                str(payload["start"]),
                "--end",
                str(payload["end"]),
                "--result",
                str(result_path),
            ]
        )
        if payload.get("ts_codes"):
            command.extend(["--ts-code", ",".join(payload["ts_codes"])])
        if payload.get("categories"):
            command.extend(["--category", ",".join(payload["categories"])])
        if int(payload.get("limit") or 0) > 0:
            command.extend(["--limit", str(payload["limit"])])
        command.extend(
            [
                "--batch-size",
                str(int(payload.get("batch_size") or ANNOUNCEMENT_DEFAULT_BATCH_SIZE)),
                "--workers",
                str(int(payload.get("workers") or ANNOUNCEMENT_DEFAULT_WORKERS)),
            ]
        )
    elif job["kind"] == "corpus_nlp":
        command.extend(
            [
                "corpus-nlp",
                "--start",
                str(payload["start"]),
                "--end",
                str(payload["end"]),
                "--result",
                str(result_path),
            ]
        )
        if payload.get("datasets"):
            command.extend(["--dataset", ",".join(payload["datasets"])])
        if payload.get("ts_codes"):
            command.extend(["--ts-code", ",".join(payload["ts_codes"])])
        if int(payload.get("limit") or 0) > 0:
            command.extend(["--limit", str(payload["limit"])])
        command.extend(
            [
                "--batch-size",
                str(int(payload.get("batch_size") or CORPUS_DEFAULT_BATCH_SIZE)),
                "--workers",
                str(int(payload.get("workers") or CORPUS_DEFAULT_WORKERS)),
                "--major-news-per-day",
                str(
                    int(
                        payload.get("major_news_per_day")
                        if payload.get("major_news_per_day") is not None
                        else CORPUS_DEFAULT_MAJOR_NEWS_PER_DAY
                    )
                ),
                "--irm-per-instrument-day",
                str(
                    int(
                        payload.get("irm_per_instrument_day")
                        if payload.get("irm_per_instrument_day") is not None
                        else CORPUS_DEFAULT_IRM_PER_INSTRUMENT_DAY
                    )
                ),
            ]
        )
    elif job["kind"] == "event_market_response":
        command.extend(
            [
                "event-market-response",
                "--snapshot-name",
                str(payload["snapshot_name"]),
                "--horizons",
                ",".join(str(value) for value in payload.get("horizons", [1, 3, 5, 20])),
                "--benchmark-code",
                str(payload.get("benchmark_code") or "000300.SH"),
                "--result",
                str(result_path),
            ]
        )
    elif job["kind"] == "report_rc_factors":
        command.extend(
            [
                "report-rc-factors",
                "--start",
                str(payload["start"]),
                "--end",
                str(payload["end"]),
                "--result",
                str(result_path),
            ]
        )
        if payload.get("ts_codes"):
            command.extend(["--ts-code", ",".join(payload["ts_codes"])])
    elif job["kind"] == "major_news_mentions":
        command.extend(
            [
                "major-news-mentions",
                "--start",
                str(payload["start"]),
                "--end",
                str(payload["end"]),
                "--result",
                str(result_path),
            ]
        )
        if payload.get("ts_codes"):
            command.extend(["--ts-code", ",".join(payload["ts_codes"])])
    else:
        command.extend(
            [
                "news-flash-factors",
                "--start",
                str(payload["start"]),
                "--end",
                str(payload["end"]),
                "--result",
                str(result_path),
            ]
        )
    return command, result_path, {}


def factor_register_command(
    worker, job: dict
) -> tuple[list[str], Path | None, dict[str, str]]:
    payload = job["payload"]
    command = [sys.executable, "-m", "quant_platform.db_cli"]
    registration_commands = {
        "announcement_factor_register": "register-announcement-factor",
        "corpus_factor_register": "register-corpus-factor",
        "report_rc_factor_register": "register-report-rc-factor",
        "major_news_mentions_factor_register": ("register-major-news-mentions-factor"),
        "news_flash_factor_register": "register-news-flash-factor",
    }
    command.append(registration_commands[job["kind"]])
    if job["kind"] != "news_flash_factor_register":
        command.extend(["--factor-name", str(payload.get("factor_name") or "all")])
    command.extend(
        [
            "--actor",
            str(payload.get("actor") or "information-pipeline-worker"),
        ]
    )
    return command, None, {}


def intraday_download_command(
    worker, job: dict
) -> tuple[list[str], Path | None, dict[str, str]]:
    payload = job["payload"]
    stored = worker.runtime_secrets.get("tushare")
    api_url = (stored or {}).get("api_url") or worker.settings.api_url
    token = (stored or {}).get("token") or worker.settings.token
    if not api_url or not token:
        raise ValueError("Tushare credentials are not configured")
    output = worker.settings.data_root / "artifacts" / "execution-data" / job["id"]
    result_path = output / "result.json"
    command = [sys.executable, "-m", "quant_data.cli"]
    if job["kind"] == "margin_eligibility_download":
        command.extend(
            [
                "margin-eligibility",
                "--start",
                payload["start"],
                "--end",
                payload["end"],
                "--result",
                str(result_path),
            ]
        )
    elif job["kind"] == "core_intraday_download":
        command.extend(
            [
                "core-intraday",
                "--start",
                payload["start"],
                "--end",
                payload["end"],
                "--snapshot-name",
                payload["snapshot_name"],
                "--result",
                str(result_path),
            ]
        )
        command.extend(["--source-lineage-id", str(payload["source_lineage_id"])])
        if payload.get("daily_dataset"):
            command.extend(["--daily-source-dataset", str(payload["daily_dataset"])])
        for option, key in (
            ("--etfs", "etfs"),
            ("--stocks", "stocks"),
            ("--indices", "indices"),
            ("--futures", "futures"),
            ("--options", "options"),
        ):
            values = payload.get(key) or []
            if values:
                command.extend([option, ",".join(values)])
        if payload.get("auto_select", False):
            command.extend(
                [
                    "--auto-universe",
                    "--max-stocks",
                    str(payload.get("max_stocks", 100)),
                    "--max-options",
                    str(payload.get("max_options", 100)),
                    "--etf-categories",
                    ",".join(
                        payload.get("etf_categories")
                        or ["broad", "industry", "gold", "bond"]
                    ),
                ]
            )
    elif job["kind"] == "ashare_5m_download":
        command.extend(
            [
                "ashare-5m",
                "--start",
                payload["start"],
                "--end",
                payload["end"],
                "--snapshot-name",
                payload["snapshot_name"],
                "--result",
                str(result_path),
            ]
        )
        source_lineage_id = str(payload.get("source_lineage_id") or "")
        if not source_lineage_id:
            raise ValueError(
                "A-share five-minute download requires a bound daily source lineage"
            )
        command.extend(["--source-lineage-id", source_lineage_id])
        if payload.get("daily_dataset"):
            command.extend(["--daily-source-dataset", str(payload["daily_dataset"])])
    elif (
        job["kind"] == "supplemental_research_corpus"
        and payload.get("profile") == "research-assets"
    ):
        command.extend(
            [
                "research-report-download",
                "--start",
                payload["start"],
                "--end",
                payload["end"],
                "--result",
                str(result_path),
            ]
        )
    else:
        command.extend(
            [
                "supplemental-download",
                "--bundle",
                payload["bundle"],
                "--start",
                payload["start"],
                "--end",
                payload["end"],
                "--result",
                str(result_path),
            ]
        )
        if payload.get("symbols"):
            command.extend(["--symbols", ",".join(payload["symbols"])])
    return command, result_path, {"TUSHARE_API_URL": api_url, "TUSHARE_TOKEN": token}


def data_verify_command(
    worker, job: dict
) -> tuple[list[str], Path | None, dict[str, str]]:
    payload = job["payload"]
    return (
        [
            sys.executable,
            "-m",
            "quant_data.cli",
            "verify",
            "--snapshot-end",
            str(payload.get("end") or "latest"),
            "--profile",
            str(payload.get("profile") or "full"),
        ],
        None,
        {},
    )


def data_snapshot_command(
    worker, job: dict
) -> tuple[list[str], Path | None, dict[str, str]]:
    payload = job["payload"]
    command = [
        sys.executable,
        "-m",
        "quant_data.cli",
        "snapshot",
        "--name",
        payload["snapshot_name"],
        "--start",
        payload["start"],
        "--end",
        payload["end"],
        "--profile",
        payload["profile"],
    ]
    if payload.get("industry_history_anchor"):
        command.extend(
            [
                "--industry-history-anchor",
                str(payload["industry_history_anchor"]),
            ]
        )
    return command, None, {}


def data_qlib_command(
    worker, job: dict
) -> tuple[list[str], Path | None, dict[str, str]]:
    payload = job["payload"]
    result_path = worker.settings.data_root / "artifacts" / "data-qlib" / job["id"] / "result.json"
    return (
        [
            sys.executable,
            "-m",
            "quant_data.cli",
            "build-qlib",
            "--snapshot",
            payload["snapshot_name"],
            "--result",
            str(result_path),
        ],
        result_path,
        {},
    )


def minute_qlib_command(
    worker, job: dict
) -> tuple[list[str], Path | None, dict[str, str]]:
    payload = job["payload"]
    command = [
        sys.executable,
        "-m",
        "quant_data.cli",
        "build-minute-qlib",
        "--snapshot",
        payload["snapshot_name"],
        "--output-name",
        payload["output_name"],
    ]
    if payload.get("target_frequency"):
        command.extend(["--target-frequency", str(payload["target_frequency"])])
    expected_manifest_sha256 = str(payload.get("snapshot_manifest_sha256") or "")
    if not expected_manifest_sha256:
        raise ValueError("minute Qlib job has no sealed snapshot manifest digest")
    command.extend(["--expected-manifest-sha256", expected_manifest_sha256])
    return command, None, {}


def bootstrap_command(
    worker, job: dict
) -> tuple[list[str], Path | None, dict[str, str]]:
    payload = job["payload"]
    stored = worker.runtime_secrets.get("tushare")
    api_url = (stored or {}).get("api_url") or worker.settings.api_url
    token = (stored or {}).get("token") or worker.settings.token
    if not api_url or not token:
        raise ValueError("Tushare credentials are not configured")
    command = [
        sys.executable,
        "-m",
        "quant_data.cli",
        "bootstrap",
        "--profile",
        payload["profile"],
        "--start",
        payload["start"],
        "--end",
        payload.get("snapshot_end") or payload["end"],
    ]
    # Worker bootstrap jobs always stop after durable download units;
    # make that contract explicit so core/research profiles do not
    # inherit the CLI's full-publication Qlib default.
    command.extend(["--download-only", "--no-build-qlib"])
    if payload.get("incremental") is True:
        command.append("--incremental")
    return (
        command,
        None,
        {
            "TUSHARE_API_URL": api_url,
            "TUSHARE_TOKEN": token,
        },
    )


COMMANDS = {
    "research_asset_acquire": research_asset_acquire_command,
    "baostock_overlap_validation": baostock_overlap_validation_command,
    "legacy_market_backfill": legacy_market_backfill_command,
    "cninfo_announcements_download": cninfo_announcements_download_command,
    "announcement_nlp": nlp_download_command,
    "corpus_nlp": nlp_download_command,
    "event_market_response": nlp_download_command,
    "report_rc_factors": nlp_download_command,
    "major_news_mentions": nlp_download_command,
    "news_flash_factors": nlp_download_command,
    "announcement_factor_register": factor_register_command,
    "corpus_factor_register": factor_register_command,
    "report_rc_factor_register": factor_register_command,
    "major_news_mentions_factor_register": factor_register_command,
    "news_flash_factor_register": factor_register_command,
    "margin_eligibility_download": intraday_download_command,
    "core_intraday_download": intraday_download_command,
    "ashare_5m_download": intraday_download_command,
    "data_verify": data_verify_command,
    "data_snapshot": data_snapshot_command,
    "data_qlib": data_qlib_command,
    "minute_qlib": minute_qlib_command,
    "bootstrap": bootstrap_command,
}

SUPPLEMENTAL_PREFIX = "supplemental_"
