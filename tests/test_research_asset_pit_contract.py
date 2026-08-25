from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from quant_data.config import Settings
from quant_data.research_assets import (
    ResearchAssetCandidate,
    VerifiedPdf,
    materialize_research_asset,
)
from quant_data.snapshot_lineage import make_lineage_id
from quant_platform.api import DataFinalizeRequest, SupplementalDownloadRequest
from quant_platform.information_schedule import (
    latest_verified_research_asset_snapshot,
)
from quant_platform.rdagent_scenarios import (
    get_rdagent_scenario,
    resolve_rdagent_assets,
)

pytestmark = pytest.mark.no_database

_PDF = b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n%%EOF\n"


def _pdf(url: str) -> VerifiedPdf:
    return VerifiedPdf(
        requested_url=url,
        final_url=url,
        redirect_chain=(),
        media_type="application/pdf",
        body=_PDF,
        sha256=hashlib.sha256(_PDF).hexdigest(),
    )


def test_report_asset_is_bound_to_pre_final_cutoff_for_explicit_and_auto_selection(
    tmp_path: Path,
) -> None:
    available_at = datetime(2026, 8, 14, 8, tzinfo=UTC)
    candidate = ResearchAssetCandidate(
        source_kind="tushare_research_report",
        source_id="https://example.com/report.pdf",
        title="Future report",
        pdf_url="https://example.com/report.pdf",
        published_at=available_at,
        available_at=available_at,
        asset_type="research_report",
        selection_as_of=available_at.date(),
    )
    published = materialize_research_asset(
        tmp_path,
        candidate,
        _pdf(candidate.pdf_url),
        acquired_at=available_at,
    )
    settings = Settings(api_url="", token="", data_root=tmp_path)
    scenario = get_rdagent_scenario("fin_factor_report")

    with pytest.raises(ValueError, match="pre-final end date"):
        resolve_rdagent_assets(settings, scenario, [published.asset_id])
    with pytest.raises(ValueError, match="became available after"):
        resolve_rdagent_assets(
            settings,
            scenario,
            [published.asset_id],
            pre_final_end=date(2026, 8, 13),
        )
    with pytest.raises(ValueError, match="requires 1 to 20"):
        resolve_rdagent_assets(
            settings,
            scenario,
            [],
            pre_final_end=date(2026, 8, 13),
        )
    resolved = resolve_rdagent_assets(
        settings,
        scenario,
        [published.asset_id],
        pre_final_end=date(2026, 8, 14),
    )
    assert resolved["assets"][0]["available_at"] == available_at.isoformat()


def _snapshot(
    data_root: Path,
    name: str,
    *,
    profile: str,
    end: str,
    datasets: tuple[str, ...],
) -> None:
    root = data_root / "snapshots" / name
    root.mkdir(parents=True)
    dataset_entries: dict[str, dict[str, object]] = {}
    for dataset in datasets:
        source = root / f"{dataset}.parquet"
        source.write_bytes(f"sealed-{dataset}".encode())
        dataset_entries[dataset] = {
            "files": [
                {
                    "path": source.name,
                    "bytes": source.stat().st_size,
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                }
            ]
        }
    lineage_configuration = {
        "profile": profile,
        "start_date": "2017-01-01",
    }
    lineage_kind = (
        "research_asset_source" if profile == "research-assets" else "qlib_daily_source"
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "profile": profile,
                "lineage_id": make_lineage_id(lineage_kind, lineage_configuration),
                "lineage_contract": {
                    "kind": lineage_kind,
                    "configuration": lineage_configuration,
                },
                "lineage_generation": 0,
                "parent_snapshot": None,
                "parent_manifest_sha256": None,
                "start_date": "2017-01-01",
                "end_date": end,
                "quality_gate": {"ok": True},
                "datasets": dataset_entries,
            }
        ),
        encoding="utf-8",
    )
    (root / "verification.json").write_text(
        json.dumps({"ok": True, "errors": []}), encoding="utf-8"
    )


def test_research_asset_snapshot_selector_rejects_newer_wrong_or_incomplete_profiles(
    tmp_path: Path,
) -> None:
    _snapshot(
        tmp_path,
        "research-source",
        profile="research-assets",
        end="2026-08-13",
        datasets=("trade_cal", "research_report"),
    )
    _snapshot(
        tmp_path,
        "newer-full",
        profile="full",
        end="2026-08-14",
        datasets=("trade_cal",),
    )
    _snapshot(
        tmp_path,
        "newer-incomplete-research",
        profile="research-assets",
        end="2026-08-14",
        datasets=("trade_cal",),
    )

    assert (
        latest_verified_research_asset_snapshot(tmp_path, as_of=date(2026, 8, 14))
        == "research-source"
    )


def test_api_exposes_only_the_isolated_research_asset_publication_chain() -> None:
    assert DataFinalizeRequest.model_validate(
        {"profile": "research-assets", "start": "2017-01-01"}
    ).profile == "research-assets"
    request = SupplementalDownloadRequest.model_validate(
        {
            "bundle": "research_corpus",
            "start": "2017-01-01",
            "publish_research_assets": True,
        }
    )
    assert request.publish_research_assets is True
    with pytest.raises(ValueError, match="only for the research_corpus"):
        SupplementalDownloadRequest.model_validate(
            {"bundle": "cn_macro", "publish_research_assets": True}
        )
