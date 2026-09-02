from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from quant_platform.announcement_nlp import (
    DEFAULT_BATCH_SIZE as ANNOUNCEMENT_DEFAULT_BATCH_SIZE,
)
from quant_platform.announcement_nlp import DEFAULT_WORKERS as ANNOUNCEMENT_DEFAULT_WORKERS
from quant_platform.announcement_nlp import MAX_BATCH_SIZE as ANNOUNCEMENT_MAX_BATCH_SIZE
from quant_platform.announcement_nlp import MAX_WORKERS as ANNOUNCEMENT_MAX_WORKERS
from quant_platform.announcement_nlp import process_announcements
from quant_platform.corpus_nlp import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CORPUS_DATASETS,
    DEFAULT_IRM_PER_INSTRUMENT_DAY,
    DEFAULT_MAJOR_NEWS_PER_DAY,
    DEFAULT_WORKERS,
    MAX_WORKERS,
    SUPPORTED_CORPUS_DATASETS,
    process_corpus,
)
from quant_platform.event_market_response import (
    DEFAULT_BENCHMARK,
    DEFAULT_HORIZONS,
    process_event_market_response,
)
from quant_platform.global_reference_factors import process_global_reference
from quant_platform.major_news_mentions import process_major_news_mentions
from quant_platform.news_flash_factors import process_news_flash
from quant_platform.report_rc_factors import process_report_rc
from quant_platform.runtime_secret_store import RuntimeSecretStore

from .baostock_provider import BAOSTOCK_SOURCE_VERSION, BaoStockProvider
from .catalog import (
    CORE_DAILY,
    CORPORATE_EVENTS,
    ETF_DAILY,
    FUNDAMENTALS,
    GLOBAL_REFERENCE_DATASETS,
    REFERENCE_FIELDS,
    RESEARCH_DAILY,
)
from .checkpoint import CheckpointStore
from .cninfo_announcements import audit_cninfo_announcements, download_cninfo_announcements
from .config import Settings
from .coverage_data import coverage_secondary_specs
from .execution_contract import (
    MINUTE_EXECUTION_CONTRACT_VERSION,
    require_daily_qlib_contract,
)
from .execution_data import MARGIN_DATASET, margin_specs
from .legacy_market import (
    BAOSTOCK_OVERLAP_POLICY_VERSION,
    DEFAULT_OVERLAP_SYMBOLS,
    LEGACY_MARKET_DATASETS,
    PRIMARY_OVERLAP_PROVIDER,
    baostock_history_specs,
    baostock_reference_specs,
    planned_baostock_universe,
    require_audited_overlap_symbols,
    require_current_primary_overlap_evidence,
    validate_baostock_overlap,
)
from .minute_qlib_builder import MinuteQlibBuilder
from .models import FetchSpec
from .partitioning import (
    is_adaptive_partition,
    is_partition_overflow_error,
    partition_bounds,
    resize_partition_spec,
    split_partition_spec,
)
from .planner import BootstrapPlanner, ExecutionDataPlanner, compact_date, parse_date, today_cn
from .provider import TushareHttpProvider
from .qlib_builder import (
    DAILY_QLIB_DUMP_WORKERS,
    QlibBuilder,
    verify_qlib_output_manifest,
)
from .rate_limit import GlobalRateGate
from .reference_data import (
    STK_SURV_PROVIDER_PAGE_LIMIT,
    select_current_reference_units,
)
from .release_window import (
    QLIB_DAILY_REQUIRED_DATASETS,
    QLIB_RESEARCH_REQUIRED_DATASETS,
    select_release_window_units,
    summarize_release_plan,
)
from .research_assets import (
    ResearchAssetError,
    acquire_manual_https_pdf,
    ingest_research_assets,
)
from .runner import DownloadRunner
from .snapshot_lineage import (
    canonical_sha256,
    file_contract_sha256,
    make_lineage_id,
    prepare_lineage_metadata,
    resolve_verified_snapshot_anchor,
    verify_snapshot_lineage,
)
from .storage import ParquetStore
from .supplemental_data import (
    SHARE_FLOAT_PROVIDER_OFFSET_CAP,
    SUPPORTED_BUNDLES,
    a_share_bulk_history_specs,
    bond_reference_specs,
    etf_constituent_history_specs,
    etf_constituent_overflow_repartition_specs,
    market_daily_specs,
    market_financial_specs,
    next_pagination_specs,
    pagination_extension_spec,
    require_pagination_terminated,
    share_float_overflow_repartition_specs,
    supplemental_specs,
    tdx_member_overflow_repartition_specs,
)
from .universe import select_intraday_universe
from .verify import (
    quality_gate_payload,
    verify_ashare_5m_source_files,
    verify_downloads,
    write_report,
)

app = typer.Typer(no_args_is_help=True, help="Resumable Tushare-to-Parquet bootstrap pipeline")
console = Console()

EXECUTION_SNAPSHOT_CONTRACT_VERSION = (
    "execution-snapshot-v3-daily-source-universe-bound"
)
QLIB_SNAPSHOT_PROFILES = frozenset({"core", "research", "full"})
RESEARCH_ASSET_SNAPSHOT_PROFILE = "research-assets"
# Isolated peripheral-markets snapshot (US/HK dailies, global indexes, US
# treasury yields, US/HK trade calendars). Never joins the A-share profiles or
# feeds the Qlib builder.
GLOBAL_REFERENCE_SNAPSHOT_PROFILE = "global-reference"
SNAPSHOT_PROFILES = frozenset(
    {
        *QLIB_SNAPSHOT_PROFILES,
        RESEARCH_ASSET_SNAPSHOT_PROFILE,
        GLOBAL_REFERENCE_SNAPSHOT_PROFILE,
    }
)
INDUSTRY_HISTORY_CARRY_RULE_VERSION = "index-member-delisted-parent-carry-v1"


# Provider probes on 2026-08-04 proved that these non-adaptive interfaces can
# legitimately exceed their original page ceilings.  Extend only capped page
# groups so their already-completed checkpoint unit keys remain reusable.
_PAGINATION_EXTENSION_MAX_PAGES = {
    "ccass_hold_detail": 128,
    "dc_member": 32,
    # Production dates can exceed the original eight 100-row pages. Continue
    # from the existing offset so completed calendar pages stay reusable while
    # a short page still proves that the date is complete.
    "eco_cal": 32,
    "tdx_member": 48,
    "ths_member": 96,
}


class ExecutionProgressReporter:
    """Write atomic live progress snapshots for the durable job worker."""

    def __init__(self, path: Path | None, target: dict[str, Any] | None = None) -> None:
        self.path = path
        self.target = dict(target or {})
        self._lock = threading.Lock()
        self._last_write = 0.0

    def set_target(self, **values: Any) -> None:
        self.target.update({key: value for key, value in values.items() if value is not None})

    def publish(
        self,
        context: Context,
        *,
        execution_phase: str,
        phase_label: str,
        datasets: set[str],
        force: bool = False,
    ) -> None:
        if self.path is None:
            return
        with self._lock:
            current = time.monotonic()
            if not force and current - self._last_write < 2.0:
                return
            checkpoint = context.checkpoint.progress_summary(datasets)
            next_retry_at = checkpoint.get("next_retry_at")
            if isinstance(next_retry_at, datetime):
                checkpoint["next_retry_at"] = next_retry_at.isoformat()
            payload = {
                "status": "running",
                "execution_phase": execution_phase,
                "phase_label": phase_label,
                "datasets": sorted(datasets),
                "target": self.target,
                "checkpoint": checkpoint,
                "updated_at": datetime.now(UTC).isoformat(),
            }
            _write_optional_result(self.path, payload)
            self._last_write = current


class Context:
    def __init__(
        self,
        settings: Settings,
        on_result=None,
        *,
        progress_path: Path | None = None,
        progress_target: dict[str, Any] | None = None,
    ) -> None:
        self.settings = settings
        self.checkpoint = CheckpointStore(settings.database_url)
        self.storage = ParquetStore(settings.data_root, keep_raw=settings.keep_raw)
        self.planner = BootstrapPlanner(self.checkpoint, self.storage)
        self.execution_planner = ExecutionDataPlanner(self.checkpoint)
        self.rate_gate = GlobalRateGate(settings.requests_per_minute)
        self.provider = TushareHttpProvider(
            api_url=settings.api_url,
            token=settings.token,
            rate_gate=self.rate_gate,
            timeout_seconds=settings.timeout_seconds,
            max_attempts=settings.max_request_attempts,
            cooldown_seconds=settings.cooldown_seconds,
        )
        self.runner = DownloadRunner(
            checkpoint=self.checkpoint,
            storage=self.storage,
            provider=self.provider,
            workers=settings.workers,
            on_result=on_result,
        )
        self.progress = ExecutionProgressReporter(progress_path, progress_target)

    def report_progress(
        self,
        execution_phase: str,
        phase_label: str,
        datasets: set[str],
        *,
        force: bool = False,
    ) -> None:
        self.progress.publish(
            self,
            execution_phase=execution_phase,
            phase_label=phase_label,
            datasets=datasets,
            force=force,
        )


def load_context(
    *,
    require_credentials: bool = True,
    on_result=None,
    progress_path: Path | None = None,
    progress_target: dict[str, Any] | None = None,
) -> Context:
    settings = Settings.from_env()
    if require_credentials:
        settings.require_credentials()
    settings.data_root.mkdir(parents=True, exist_ok=True)
    return Context(
        settings,
        on_result=on_result,
        progress_path=progress_path,
        progress_target=progress_target,
    )


def _phase_for_label(label: str) -> str:
    normalized = label.lower()
    if (
        "overflow continuation" in normalized
        or "partition continuation" in normalized
        or "adaptive takeover" in normalized
    ):
        return "adaptive_recovery"
    if "pagination" in normalized:
        return "pagination"
    if "calendar" in normalized or "master" in normalized or "basic" in normalized:
        return "prerequisites"
    return "downloading"


def _run_phase(context: Context, label: str, datasets: set[str]) -> None:
    total = context.checkpoint.remaining_count(datasets)
    if total == 0:
        console.print(f"[green]{label}: already complete[/green]")
        context.report_progress(_phase_for_label(label), label, datasets, force=True)
        return
    context.report_progress(_phase_for_label(label), label, datasets, force=True)
    with Progress(console=console) as progress:
        task = progress.add_task(label, total=total)
        previous_on_result = context.runner.on_result

        def on_result(dataset: str, succeeded: bool, rows: int) -> None:
            progress.advance(task)
            context.report_progress(_phase_for_label(label), label, datasets)
            if previous_on_result:
                previous_on_result(dataset, succeeded, rows)

        context.runner.on_result = on_result
        try:
            summary = context.runner.run(datasets)
        finally:
            context.runner.on_result = previous_on_result
    context.report_progress(_phase_for_label(label), label, datasets, force=True)
    console.print(
        f"{label}: succeeded={summary.succeeded} failed={summary.failed} rows={summary.rows}"
    )


def _require_selected_plan_complete(
    context: Context,
    datasets: set[str],
    *,
    label: str,
    snapshot_start: date,
    snapshot_end: date,
    required_datasets: set[str] | frozenset[str] | None = None,
) -> None:
    """Fail closed when a selected release-window work unit is incomplete."""

    selection = select_release_window_units(
        context.checkpoint.active_units(datasets),
        snapshot_start=snapshot_start,
        snapshot_end=snapshot_end,
        datasets=datasets,
    )
    rows = summarize_release_plan(selection.rows)
    observed = {str(row["dataset"]) for row in rows}
    missing = sorted(set(required_datasets or datasets) - observed)
    if missing:
        console.print(
            f"[red]{label} has no active plan for: {', '.join(missing)}[/red]"
        )
        raise typer.Exit(2)
    incomplete = [
        row for row in rows if int(row["succeeded"] or 0) != int(row["planned"] or 0)
    ]
    if not incomplete:
        return
    remaining = sum(
        max(0, int(row["planned"] or 0) - int(row["succeeded"] or 0))
        for row in incomplete
    )
    console.print(
        f"[red]{label} incomplete; refusing to continue[/red]: "
        f"remaining={remaining} datasets={len(incomplete)}"
    )
    for row in incomplete[:20]:
        console.print(
            "  - "
            f"{row['dataset']}: succeeded={int(row['succeeded'] or 0)}/"
            f"planned={int(row['planned'] or 0)}, failed={int(row['failed'] or 0)}, "
            f"running={int(row['running'] or 0)}"
        )
    raise typer.Exit(2)


def _run_paginated_specs(
    context: Context, label: str, initial_specs: list[FetchSpec]
) -> tuple[list[FetchSpec], list[dict], int]:
    if not initial_specs:
        return [], [], 0
    initial_datasets = {spec.dataset for spec in initial_specs}
    context.report_progress("planning", f"{label} planning", initial_datasets, force=True)
    specs = _reconcile_range_plan(context, list(initial_specs))
    if not specs:
        return [], [], 0
    specs, inserted = _activate_stk_surv_plan(context, specs)
    specs = _rehydrate_durable_pagination_specs(context, specs)
    datasets = {spec.dataset for spec in specs}
    ignored_keys: set[str] = set()
    recovery_specs, recovered_keys = _pagination_overflow_recovery(context, specs, ignored_keys)
    if recovered_keys:
        ignored_keys.update(recovered_keys)
        known = {spec.unit_key for spec in specs}
        recovery_specs = [spec for spec in recovery_specs if spec.unit_key not in known]
        specs.extend(recovery_specs)
        inserted += context.checkpoint.add(recovery_specs)
        specs = _rehydrate_durable_pagination_specs(context, specs)
    _supersede_unplanned_range_units(context, specs)
    _run_phase(context, label, datasets)

    while True:
        recovery_specs, recovered_keys = _pagination_overflow_recovery(context, specs, ignored_keys)
        if recovered_keys:
            ignored_keys.update(recovered_keys)
            known = {spec.unit_key for spec in specs}
            recovery_specs = [spec for spec in recovery_specs if spec.unit_key not in known]
            specs.extend(recovery_specs)
            inserted += context.checkpoint.add(recovery_specs)
            specs = _rehydrate_durable_pagination_specs(context, specs)
            _run_phase(context, f"{label} overflow continuation", datasets)
            continue

        active_specs = _exclude_superseded_specs(
            context,
            [spec for spec in specs if spec.unit_key not in ignored_keys],
        )
        rows = _require_specs_complete(context, active_specs)
        next_specs = next_pagination_specs(active_specs, rows)
        next_specs, takeover_specs, takeover_ignored_keys = (
            _reconcile_superseded_next_specs(context, active_specs, next_specs)
        )
        if takeover_ignored_keys:
            ignored_keys.update(takeover_ignored_keys)
            known = {spec.unit_key for spec in specs}
            takeover_specs = [
                spec for spec in takeover_specs if spec.unit_key not in known
            ]
            specs.extend(takeover_specs)
            inserted += context.checkpoint.add(takeover_specs)
            specs = _rehydrate_durable_pagination_specs(context, specs)
            _run_phase(context, f"{label} adaptive takeover", datasets)
            continue
        if not next_specs:
            recovery_specs, recovered_keys = _full_page_partition_recovery(
                context, active_specs, rows, ignored_keys
            )
            if recovered_keys:
                known = {spec.unit_key for spec in specs}
                recovery_specs = [spec for spec in recovery_specs if spec.unit_key not in known]
                specs.extend(recovery_specs)
                inserted += context.checkpoint.add(recovery_specs)
                specs = _rehydrate_durable_pagination_specs(context, specs)
                _run_phase(context, f"{label} partition continuation", datasets)
                continue
            require_pagination_terminated(active_specs, rows)
            context.report_progress(
                "verifying", f"{label} pagination verified", datasets, force=True
            )
            return specs, rows, inserted

        specs.extend(next_specs)
        specs = _rehydrate_durable_pagination_specs(context, specs)
        recovery_specs, recovered_keys = _pagination_overflow_recovery(context, specs, ignored_keys)
        if recovered_keys:
            ignored_keys.update(recovered_keys)
            known = {spec.unit_key for spec in specs}
            recovery_specs = [spec for spec in recovery_specs if spec.unit_key not in known]
            specs.extend(recovery_specs)
            inserted += context.checkpoint.add(recovery_specs)
            specs = _rehydrate_durable_pagination_specs(context, specs)
            _run_phase(context, f"{label} overflow continuation", datasets)
            continue

        inserted += context.checkpoint.add(next_specs)
        _run_phase(context, f"{label} pagination", datasets)


_RANGE_REUSE_DATASETS = {
    "fund_share",
    "moneyflow_hsgt",
    "moneyflow_cnt_ths",
    "moneyflow_ind_ths",
    "moneyflow_ind_dc",
    "moneyflow_mkt_dc",
    "etf_sh_cons",
    "etf_sz_cons",
    "cyq_perf",
    "cyq_chips",
}


def _activate_stk_surv_plan(
    context: Context, specs: list[FetchSpec]
) -> tuple[list[FetchSpec], int]:
    """Durably add replacements before retiring their legacy request units."""

    reconciled, obsolete = _reconcile_stk_surv_plan(context, specs)
    inserted = context.checkpoint.add(reconciled)
    if not obsolete:
        return reconciled, inserted
    replacement_rows = context.checkpoint.unit_rows(set(obsolete.values()))
    durable_replacements = {
        str(row["unit_key"])
        for row in replacement_rows
        if str(row.get("status") or "") != "superseded"
    }
    retired = [
        legacy_key
        for legacy_key, replacement_key in obsolete.items()
        if replacement_key in durable_replacements
    ]
    context.checkpoint.supersede_units(
        retired,
        "legacy unpaged stk_surv unit superseded by explicit 400-row pagination",
    )
    return reconciled, inserted


def _reconcile_stk_surv_plan(
    context: Context, specs: list[FetchSpec]
) -> tuple[list[FetchSpec], dict[str, str]]:
    """Migrate the legacy unpaged survey contract without losing good work.

    A legacy single-day response below 400 rows proves termination and can be
    reused verbatim. A 400-row success is truncated and is replaced by the new
    explicit page group. Unfinished legacy units are retained as superseded
    audit rows so the runner cannot retry the obsolete request contract.
    """

    targets = [spec for spec in specs if spec.dataset == "stk_surv"]
    if not targets:
        return specs, {}
    target_keys = {spec.unit_key for spec in targets}
    existing_current = {
        str(row["unit_key"])
        for row in context.checkpoint.unit_rows(target_keys)
        if str(row.get("status") or "") != "superseded"
    }
    target_identities = {_stk_surv_day_identity(spec) for spec in targets}
    reusable: dict[tuple[str, str], FetchSpec] = {}
    for row in context.checkpoint.successful("stk_surv"):
        candidate = _checkpoint_row_spec(row)
        identity = _stk_surv_day_identity(candidate)
        row_count = row.get("row_count")
        if (
            identity in target_identities
            and _is_legacy_stk_surv_spec(candidate)
            and row_count is not None
            and 0 <= int(row_count) < STK_SURV_PROVIDER_PAGE_LIMIT
        ):
            reusable[identity] = candidate

    reconciled: list[FetchSpec] = []
    for spec in specs:
        if spec.dataset != "stk_surv" or spec.unit_key in existing_current:
            reconciled.append(spec)
            continue
        reconciled.append(reusable.get(_stk_surv_day_identity(spec), spec))

    replacements = {
        _stk_surv_day_identity(spec): spec.unit_key
        for spec in reconciled
        if spec.dataset == "stk_surv" and not _is_legacy_stk_surv_spec(spec)
    }
    obsolete: dict[str, str] = {}
    for row in context.checkpoint.unfinished_units("stk_surv"):
        candidate = _checkpoint_row_spec(row)
        replacement_key = replacements.get(_stk_surv_day_identity(candidate))
        if _is_legacy_stk_surv_spec(candidate) and replacement_key:
            obsolete[candidate.unit_key] = replacement_key
    return reconciled, obsolete


def _stk_surv_day_identity(spec: FetchSpec) -> tuple[str, str]:
    return (
        str(spec.params.get("start_date") or ""),
        str(spec.params.get("end_date") or ""),
    )


def _is_legacy_stk_surv_spec(spec: FetchSpec) -> bool:
    return (
        spec.dataset == "stk_surv"
        and "page_group" not in spec.scope
        and "limit" not in spec.params
        and "offset" not in spec.params
        and int(spec.scope.get("row_limit") or 0) == STK_SURV_PROVIDER_PAGE_LIMIT
    )


def _reconcile_range_plan(context: Context, specs: list[FetchSpec]) -> list[FetchSpec]:
    """Reuse complete legacy partitions and plan only uncovered session gaps."""

    targets: list[FetchSpec] = []
    untouched: list[FetchSpec] = []
    for spec in specs:
        if spec.dataset in _RANGE_REUSE_DATASETS and is_adaptive_partition(spec):
            targets.append(spec)
        else:
            untouched.append(spec)
    if not targets:
        return specs
    # An exact successful page-zero partition is sufficient to resume the
    # same frozen plan. The pagination rehydration step restores its durable
    # siblings and the normal termination check still fails closed if the
    # group is incomplete. Avoid decoding hundreds of thousands of historical
    # ETF pages merely to rediscover an unchanged request window.
    exact_rows = {
        str(row["unit_key"]): row
        for row in context.checkpoint.unit_rows(spec.unit_key for spec in targets)
    }
    if all(
        str((exact_rows.get(spec.unit_key) or {}).get("status")) == "succeeded"
        for spec in targets
    ):
        return specs
    rows_by_dataset = {
        dataset: context.checkpoint.successful(dataset)
        for dataset in {spec.dataset for spec in targets}
    }
    success_index: dict[str, dict[tuple[tuple[str, object], ...], list[FetchSpec]]] = {}
    for dataset, rows in rows_by_dataset.items():
        by_identity: dict[tuple[tuple[str, object], ...], list[FetchSpec]] = {}
        for candidate in _complete_success_specs(rows):
            identity = tuple(sorted(_partition_identity(candidate).items()))
            by_identity.setdefault(identity, []).append(candidate)
        success_index[dataset] = by_identity
    reusable: dict[str, FetchSpec] = {}
    replacements: list[FetchSpec] = []
    for target in targets:
        _, target_start, target_end = partition_bounds(target)
        assert isinstance(target_start, date) and not isinstance(target_start, datetime)
        assert isinstance(target_end, date) and not isinstance(target_end, datetime)
        target_identity = tuple(sorted(_partition_identity(target).items()))
        index = success_index[target.dataset]
        successful = [*index.get((), [])]
        if target_identity:
            successful.extend(index.get(target_identity, []))
        matching: list[FetchSpec] = []
        for candidate in successful:
            bounds = _date_partition_bounds(candidate)
            if bounds is None:
                continue
            candidate_start, candidate_end = bounds
            if target_start <= candidate_start <= candidate_end <= target_end:
                matching.append(candidate)
        if not matching:
            replacements.append(target)
            continue

        raw_values = target.scope.get("partition_values")
        values = (
            [datetime.strptime(str(value), "%Y%m%d").date() for value in raw_values]
            if isinstance(raw_values, list)
            else [
                target_start + timedelta(days=offset)
                for offset in range((target_end - target_start).days + 1)
            ]
        )
        covered: set[date] = set()
        for candidate in matching:
            reusable[candidate.unit_key] = candidate
            candidate_start, candidate_end = _date_partition_bounds(candidate) or (
                target_start,
                target_end,
            )
            covered.update(value for value in values if candidate_start <= value <= candidate_end)
        segment: list[date] = []
        for value in values:
            if value in covered:
                if segment:
                    replacements.append(resize_partition_spec(target, segment[0], segment[-1]))
                    segment = []
            else:
                segment.append(value)
        if segment:
            replacements.append(resize_partition_spec(target, segment[0], segment[-1]))

    planned = [*untouched, *reusable.values(), *replacements]
    planned_keys = {spec.unit_key for spec in planned}
    stale = []
    for dataset in {spec.dataset for spec in targets}:
        for row in context.checkpoint.unfinished_units(dataset):
            spec = _checkpoint_row_spec(row)
            if spec.unit_key not in planned_keys:
                stale.append(spec.unit_key)
    context.checkpoint.supersede_units(
        stale,
        "legacy unfinished unit superseded by adaptive range planning",
    )
    return planned


def _supersede_unplanned_range_units(context: Context, specs: list[FetchSpec]) -> int:
    """Retire unfinished range rows that are absent from the executable plan.

    Run this after overflow recovery has rehydrated every child that still
    belongs to the plan so obsolete adaptive children cannot remain runnable.
    """

    range_specs = [
        spec
        for spec in specs
        if spec.dataset in _RANGE_REUSE_DATASETS and is_adaptive_partition(spec)
    ]
    if not range_specs:
        return 0
    planned_keys = {spec.unit_key for spec in range_specs}
    stale = [
        str(row["unit_key"])
        for dataset in {spec.dataset for spec in range_specs}
        for row in context.checkpoint.unfinished_units(dataset)
        if str(row["unit_key"]) not in planned_keys
    ]
    return context.checkpoint.supersede_units(
        stale,
        "unfinished adaptive unit superseded by the executable range plan",
    )


def _exclude_superseded_specs(context: Context, specs: list[FetchSpec]) -> list[FetchSpec]:
    """Keep audit-only checkpoint rows out of pagination completeness checks."""

    superseded = context.checkpoint.superseded_unit_keys(spec.unit_key for spec in specs)
    if not superseded:
        return specs
    return [spec for spec in specs if spec.unit_key not in superseded]


def _pagination_contract(spec: FetchSpec) -> tuple[object, ...]:
    """Return the immutable request contract shared by pages in one group."""

    scope = {
        key: value
        for key, value in spec.scope.items()
        if key not in {"offset", "page_index", "max_pages"}
    }
    params = {
        key: value
        for key, value in spec.params.items()
        if key not in {"limit", "offset"}
    }
    return (
        spec.dataset,
        spec.api_name,
        scope,
        params,
        tuple(spec.fields),
        bool(spec.allow_empty),
    )


def _rehydrate_durable_pagination_specs(
    context: Context, specs: list[FetchSpec]
) -> list[FetchSpec]:
    """Attach all durable siblings for every pagination group in ``specs``.

    Pagination keys are immutable and content addressed. On restart the
    planner emits page zero again; without rehydration the loop has to discover
    page 1..N serially even though every page is already in PostgreSQL. The
    stored contract is checked against the live group before a page is reused,
    so a group-name collision fails closed instead of mixing request shapes.
    """

    group_contracts: dict[tuple[str, str], tuple[object, ...]] = {}
    for spec in specs:
        group = str(spec.scope.get("page_group") or "")
        if not group:
            continue
        identity = (spec.dataset, group)
        contract = _pagination_contract(spec)
        existing = group_contracts.setdefault(identity, contract)
        if existing != contract:
            raise RuntimeError(
                "pagination plan contains conflicting live contracts for "
                f"{spec.dataset}/{group}"
            )
    if not group_contracts:
        return specs

    known = {spec.unit_key for spec in specs}
    durable: list[FetchSpec] = []
    for row in context.checkpoint.pagination_group_units(group_contracts):
        candidate = _checkpoint_row_spec(row)
        group = str(candidate.scope.get("page_group") or "")
        identity = (candidate.dataset, group)
        expected = group_contracts.get(identity)
        if expected is None:
            continue
        if _pagination_contract(candidate) != expected:
            # Refreshable reference datasets intentionally keep a stable page
            # group across weekly generations. Those older immutable rows are
            # valid audit history, but they are not siblings of the live
            # request contract and must simply remain outside this run.
            continue
        if candidate.unit_key not in known:
            known.add(candidate.unit_key)
            durable.append(candidate)
    return [*specs, *durable]


def _superseded_page_groups(scope: dict[str, Any]) -> set[str]:
    groups: set[str] = set()
    singular = scope.get("supersedes_page_group")
    if singular:
        groups.add(str(singular))
    plural = scope.get("supersedes_page_groups")
    if isinstance(plural, (list, tuple, set, frozenset)):
        groups.update(str(value) for value in plural if value)
    return groups


def _reconcile_superseded_next_specs(
    context: Context,
    current_specs: list[FetchSpec],
    next_specs: list[FetchSpec],
) -> tuple[list[FetchSpec], list[FetchSpec], set[str]]:
    """Stop immutable superseded cursors from being regenerated forever.

    A superseded next-page key is safe to omit only when durable, active child
    units explicitly declare that they replace the whole parent page group.
    Merely continuing a group is not replacement evidence: silently dropping
    the parent in that case would discard its completed prefix.
    """

    if not next_specs:
        return [], [], set()
    rows_by_key = {
        str(row["unit_key"]): row
        for row in context.checkpoint.unit_rows(spec.unit_key for spec in next_specs)
    }
    superseded = {
        spec.unit_key
        for spec in next_specs
        if str((rows_by_key.get(spec.unit_key) or {}).get("status")) == "superseded"
    }
    if not superseded:
        return next_specs, [], set()

    durable_by_dataset = {
        dataset: context.checkpoint.dataset_units(dataset)
        for dataset in {spec.dataset for spec in next_specs if spec.unit_key in superseded}
    }
    takeovers: dict[str, FetchSpec] = {}
    ignored_keys: set[str] = set()
    taken_over_groups: set[tuple[str, str]] = set()
    unresolved: list[FetchSpec] = []
    for candidate in next_specs:
        if candidate.unit_key not in superseded:
            continue
        parent_group = str(candidate.scope.get("page_group") or "")
        parent_specs = [
            spec
            for spec in current_specs
            if spec.dataset == candidate.dataset
            and str(spec.scope.get("page_group") or "") == parent_group
        ]
        durable_takeovers = [
            row
            for row in durable_by_dataset[candidate.dataset]
            if str(row.get("status")) != "superseded"
            and parent_group
            in _superseded_page_groups(dict(row.get("scope_json") or {}))
        ]
        if not parent_group or not parent_specs or not durable_takeovers:
            unresolved.append(candidate)
            continue
        taken_over_groups.add((candidate.dataset, parent_group))
        ignored_keys.update(spec.unit_key for spec in parent_specs)
        for row in durable_takeovers:
            spec = _checkpoint_row_spec(row)
            takeovers[spec.unit_key] = spec

    if unresolved:
        preview = ", ".join(
            f"{spec.dataset}/{spec.scope.get('page_group') or spec.unit_key}"
            for spec in unresolved[:5]
        )
        raise RuntimeError(
            "superseded pagination cursor lacks durable adaptive takeover evidence: "
            f"{preview}"
        )

    ignored_keys.update(
        spec.unit_key
        for spec in next_specs
        if (spec.dataset, str(spec.scope.get("page_group") or ""))
        in taken_over_groups
    )
    runnable = [
        spec
        for spec in next_specs
        if (spec.dataset, str(spec.scope.get("page_group") or ""))
        not in taken_over_groups
    ]
    return runnable, list(takeovers.values()), ignored_keys


def _supersede_unsupported_governance_units(context: Context) -> int:
    retired_chips = [
        str(row["unit_key"]) for row in context.checkpoint.unfinished_units("cyq_chips")
    ]
    superseded = context.checkpoint.supersede_units(
        retired_chips,
        "cyq_chips raw price distribution retired from the required governance plan; "
        "cyq_perf is the canonical compact dataset",
    )
    stale: list[str] = []
    for dataset in ("ccass_hold", "ccass_hold_detail"):
        for row in context.checkpoint.unfinished_units(dataset):
            params = dict(row.get("params_json") or {})
            requested_date = str(
                params.get("trade_date") or params.get("start_date") or params.get("end_date") or ""
            ).replace("-", "")[:8]
            if requested_date and requested_date < "20160101":
                stale.append(str(row["unit_key"]))
    superseded += context.checkpoint.supersede_units(
        stale,
        "provider does not support CCASS history before 2016",
    )
    return superseded


def _supersede_unsupported_research_units(context: Context) -> int:
    """Retire provider contracts that are unavailable and unused by research."""

    stale = [
        str(row["unit_key"])
        for dataset in ("wc_list", "wc_cnt")
        for row in context.checkpoint.unfinished_units(dataset)
    ]
    return context.checkpoint.supersede_units(
        stale,
        "wc_list/wc_cnt are absent from the current provider catalog and rejected by "
        "the production gateway; retained as unavailable audit rows",
    )


def _complete_success_specs(rows: list[dict]) -> list[FetchSpec]:
    specs_by_group: dict[str, list[tuple[FetchSpec, int]]] = {}
    result: list[FetchSpec] = []
    for row in rows:
        spec = _checkpoint_row_spec(row)
        group = spec.scope.get("page_group")
        if not group:
            result.append(spec)
            continue
        specs_by_group.setdefault(str(group), []).append((spec, int(row.get("row_count") or 0)))
    for pages in specs_by_group.values():
        ordered = sorted(pages, key=lambda item: int(item[0].scope.get("offset") or 0))
        if ordered[-1][1] < int(ordered[-1][0].scope["page_size"]):
            result.extend(spec for spec, _ in ordered)
    return result


def _checkpoint_row_spec(row: dict) -> FetchSpec:
    return FetchSpec(
        dataset=str(row["dataset"]),
        api_name=str(row["api_name"]),
        scope=dict(row.get("scope_json") or {}),
        params=dict(row.get("params_json") or {}),
        fields=tuple(row.get("fields_json") or ()),
        allow_empty=bool(row.get("allow_empty")),
        max_attempts=int(row.get("max_attempts") or 1),
    )


def _partition_identity(spec: FetchSpec) -> dict[str, object]:
    ignored = {
        "start_date",
        "end_date",
        "trade_date",
        "nav_date",
        "ann_date",
        "limit",
        "offset",
    }
    return {key: value for key, value in spec.params.items() if key not in ignored}


def _date_partition_bounds(spec: FetchSpec) -> tuple[date, date] | None:
    if is_adaptive_partition(spec):
        axis, start, end = partition_bounds(spec)
        if axis == "date" and isinstance(start, date) and isinstance(end, date):
            return start, end
    params = spec.params
    start_value = params.get("start_date") or params.get("trade_date")
    end_value = params.get("end_date") or params.get("trade_date")
    if not start_value or not end_value:
        return None
    try:
        start = datetime.fromisoformat(str(start_value)).date()
        end = datetime.fromisoformat(str(end_value)).date()
    except ValueError:
        try:
            start = datetime.strptime(str(start_value)[:8], "%Y%m%d").date()
            end = datetime.strptime(str(end_value)[:8], "%Y%m%d").date()
        except ValueError:
            return None
    return start, end


def _share_float_overflow_recovery(
    context: Context,
    specs: list[FetchSpec],
    ignored_keys: set[str],
) -> tuple[list[FetchSpec], set[str]]:
    """Replace provider-capped pages with disjoint date/symbol continuations."""

    share_float_specs = [
        spec
        for spec in specs
        if spec.dataset == "share_float" and spec.unit_key not in ignored_keys
    ]
    if not share_float_specs:
        return [], set()
    rows_by_key = {
        str(row["unit_key"]): row
        for row in context.checkpoint.unit_rows(
            spec.unit_key for spec in share_float_specs
        )
    }
    recovery_candidates = [
        spec
        for spec in share_float_specs
        if str((rows_by_key.get(spec.unit_key) or {}).get("status"))
        in {"failed", "superseded"}
    ]
    if not recovery_candidates:
        return [], set()
    replacements_by_parent: dict[str, list[FetchSpec]] = {}
    for child in context.checkpoint.dataset_units("share_float"):
        if str(child.get("status")) == "superseded":
            continue
        parent_group = str(dict(child.get("scope_json") or {}).get("supersedes_page_group") or "")
        if parent_group:
            replacements_by_parent.setdefault(parent_group, []).append(_checkpoint_row_spec(child))
    recovery_specs: list[FetchSpec] = []
    recovered_keys: set[str] = set()
    stock_master: pd.DataFrame | None = None
    for failed_spec in recovery_candidates:
        row = rows_by_key.get(failed_spec.unit_key)
        if not row:
            continue
        if str(row.get("status")) == "superseded":
            # A previous run may already have replaced the unstable monthly
            # page group with immutable child partitions. Rehydrate the exact
            # durable plan (which may contain trading days only) instead of
            # inventing extra calendar-day requests on restart.
            parent_group = str(failed_spec.scope.get("page_group") or "")
            replacements = replacements_by_parent.get(parent_group, [])
            if replacements:
                recovery_specs.extend(replacements)
                recovered_keys.add(failed_spec.unit_key)
                continue
        error = str(row.get("last_error") or "")
        offset = int(failed_spec.params.get("offset") or 0)
        # Once a share_float cursor reaches the provider's documented offset
        # ceiling it is structurally unrecoverable, even if a later retry
        # overwrites the original 50101 with a transient cooldown/rate-limit
        # error.  Partition from the cursor itself instead of depending on the
        # mutable last_error string.
        if offset < 100_000:
            continue

        start_text = str(failed_spec.params.get("start_date") or "")
        end_text = str(failed_spec.params.get("end_date") or "")
        symbols: list[str] = []
        if start_text == end_text and len(start_text) == 8:
            if stock_master is None:
                stock_master = context.storage.read_units(
                    context.checkpoint.successful("stock_basic")
                )
            partition_date = datetime.strptime(start_text, "%Y%m%d").date()
            symbols = _historical_a_share_symbols(
                stock_master,
                start=partition_date,
                end=partition_date,
            )
        recovery_specs.extend(share_float_overflow_repartition_specs(failed_spec, symbols))
        recovered_keys.add(failed_spec.unit_key)
        context.checkpoint.supersede_units(
            [failed_spec.unit_key],
            f"{error}; pagination offset cap superseded by disjoint date continuations",
        )
    return recovery_specs, recovered_keys


def _pagination_overflow_recovery(
    context: Context,
    specs: list[FetchSpec],
    ignored_keys: set[str],
) -> tuple[list[FetchSpec], set[str]]:
    """Recover every provider offset cap using a smaller documented partition."""

    recovery_specs, recovered_keys = _share_float_overflow_recovery(context, specs, ignored_keys)
    tdx_specs, tdx_recovered = _tdx_member_overflow_recovery(
        context, specs, ignored_keys | recovered_keys
    )
    recovery_specs.extend(tdx_specs)
    recovered_keys.update(tdx_recovered)
    rows_by_key = {
        str(row["unit_key"]): row
        for row in context.checkpoint.unit_rows(spec.unit_key for spec in specs)
    }
    for failed_spec in specs:
        if (
            failed_spec.unit_key in ignored_keys
            or failed_spec.unit_key in recovered_keys
            or not is_adaptive_partition(failed_spec)
        ):
            continue
        row = rows_by_key.get(failed_spec.unit_key)
        if not row or str(row.get("status")) not in {"failed", "superseded"}:
            continue
        error = str(row.get("last_error") or "")
        if not is_partition_overflow_error(error):
            continue
        recovery_specs.extend(split_partition_spec(failed_spec))
        recovered_keys.add(failed_spec.unit_key)
        context.checkpoint.supersede_units(
            [failed_spec.unit_key],
            f"{error}; superseded by disjoint adaptive child partitions",
        )

    etf_master: pd.DataFrame | None = None
    for failed_spec in specs:
        if (
            failed_spec.unit_key in ignored_keys
            or failed_spec.unit_key in recovered_keys
            or failed_spec.dataset not in {"etf_sh_cons", "etf_sz_cons"}
        ):
            continue
        row = rows_by_key.get(failed_spec.unit_key)
        if not row or str(row.get("status")) not in {"failed", "superseded"}:
            continue
        error = str(row.get("last_error") or "")
        offset = int(failed_spec.params.get("offset") or 0)
        normalized_error = error.replace("-", " ")
        if offset < 100_000 or ("code=50101" not in error and "offset cap" not in normalized_error):
            continue

        if (
            failed_spec.params.get("ts_code")
            and failed_spec.params.get("start_date")
            and failed_spec.params.get("end_date")
        ):
            recovery_specs.extend(etf_constituent_overflow_repartition_specs(failed_spec))
        else:
            if etf_master is None:
                etf_master = context.storage.read_units(context.checkpoint.successful("etf_basic"))
                if "ts_code" not in etf_master.columns:
                    raise RuntimeError(
                        "etf_basic did not provide ts_code for constituent overflow recovery"
                    )
            symbols = _eligible_etf_symbols(
                etf_master,
                dataset=failed_spec.dataset,
                trade_date=str(failed_spec.params["trade_date"]),
            )
            recovery_specs.extend(etf_constituent_overflow_repartition_specs(failed_spec, symbols))
        recovered_keys.add(failed_spec.unit_key)
        context.checkpoint.supersede_units(
            [failed_spec.unit_key],
            f"{error}; pagination offset cap superseded by smaller ETF partitions",
        )
    return recovery_specs, recovered_keys


def _tdx_member_overflow_recovery(
    context: Context,
    specs: list[FetchSpec],
    ignored_keys: set[str],
) -> tuple[list[FetchSpec], set[str]]:
    """Replace TDX member cursors at the provider cap with index partitions."""

    candidates = [
        spec
        for spec in specs
        if spec.unit_key not in ignored_keys
        and spec.dataset == "tdx_member"
        and int(spec.params.get("offset") or 0) >= 100_000
    ]
    if not candidates:
        return [], set()
    rows_by_key = {
        str(row["unit_key"]): row
        for row in context.checkpoint.unit_rows(spec.unit_key for spec in candidates)
    }
    failed = [
        spec
        for spec in candidates
        if str((rows_by_key.get(spec.unit_key) or {}).get("status")) in {"failed", "superseded"}
    ]
    if not failed:
        return [], set()

    index_frame = context.storage.read_units(context.checkpoint.successful("tdx_index"))
    if "ts_code" not in index_frame.columns or "trade_date" not in index_frame.columns:
        raise RuntimeError("tdx_index did not provide ts_code and trade_date for recovery")
    normalized_dates = pd.to_datetime(index_frame["trade_date"], errors="coerce").dt.strftime(
        "%Y%m%d"
    )
    recovery_specs: list[FetchSpec] = []
    recovered_keys: set[str] = set()
    for failed_spec in failed:
        trade_date = str(failed_spec.params.get("trade_date") or "")
        symbols = index_frame.loc[normalized_dates == trade_date, "ts_code"].dropna().tolist()
        recovery_specs.extend(tdx_member_overflow_repartition_specs(failed_spec, symbols))
        recovered_keys.add(failed_spec.unit_key)
        error = str((rows_by_key.get(failed_spec.unit_key) or {}).get("last_error") or "")
        context.checkpoint.supersede_units(
            [failed_spec.unit_key],
            f"{error}; pagination offset cap superseded by per-index TDX partitions",
        )
    return recovery_specs, recovered_keys


def _full_page_partition_recovery(
    context: Context,
    specs: list[FetchSpec],
    rows: list[dict],
    ignored_keys: set[str],
) -> tuple[list[FetchSpec], set[str]]:
    """Split a bisectable page group whose final allowed page is still full."""

    row_counts = {str(row["unit_key"]): int(row.get("row_count") or 0) for row in rows}
    continued: set[str] = set()
    for spec in specs:
        parent = spec.scope.get("continues_page_group") or spec.scope.get("supersedes_page_group")
        if parent:
            continued.add(str(parent))
        parents = spec.scope.get("supersedes_page_groups")
        if isinstance(parents, (list, tuple, set, frozenset)):
            continued.update(str(value) for value in parents if value)
    groups: dict[str, list[FetchSpec]] = {}
    for spec in specs:
        group = spec.scope.get("page_group")
        if group and spec.unit_key not in ignored_keys:
            groups.setdefault(str(group), []).append(spec)

    children: list[FetchSpec] = []
    recovered: set[str] = set()
    for group, pages in groups.items():
        if group in continued:
            continue
        current = max(pages, key=lambda item: int(item.scope.get("offset") or 0))
        page_size = int(current.scope["page_size"])
        page_index = int(
            current.scope.get("page_index", int(current.scope.get("offset") or 0) // page_size)
        )
        share_float_cap = (
            current.dataset == "share_float"
            and (page_index + 1) * page_size >= SHARE_FLOAT_PROVIDER_OFFSET_CAP
            and row_counts.get(current.unit_key, -1) >= page_size
        )
        if share_float_cap:
            start_text = str(current.params.get("start_date") or "")
            end_text = str(current.params.get("end_date") or "")
            symbols: list[str] = []
            if start_text == end_text and len(start_text) == 8:
                stock_master = context.storage.read_units(
                    context.checkpoint.successful("stock_basic")
                )
                partition_date = datetime.strptime(start_text, "%Y%m%d").date()
                symbols = _historical_a_share_symbols(
                    stock_master,
                    start=partition_date,
                    end=partition_date,
                )
            children.extend(share_float_overflow_repartition_specs(current, symbols))
            recovered.add(current.unit_key)
            context.checkpoint.supersede_units(
                [page.unit_key for page in pages],
                "provider offset cap avoided by disjoint share_float partitions",
            )
            continue
        extension_max_pages = _PAGINATION_EXTENSION_MAX_PAGES.get(current.dataset)
        max_pages = int(current.scope.get("max_pages") or 0)
        final_page_is_full = (
            page_index + 1 >= max_pages and row_counts.get(current.unit_key, -1) >= page_size
        )
        if extension_max_pages and final_page_is_full:
            children.append(pagination_extension_spec(current, max_pages=extension_max_pages))
            recovered.add(current.unit_key)
            continue
        if not is_adaptive_partition(current):
            continue
        max_pages = int(current.scope["max_pages"])
        if page_index + 1 < max_pages or row_counts.get(current.unit_key, -1) < page_size:
            continue
        children.extend(split_partition_spec(current))
        recovered.add(current.unit_key)
        context.checkpoint.supersede_units(
            [page.unit_key for page in pages],
            "full final pagination page superseded by disjoint adaptive child partitions",
        )
    return children, recovered


def _eligible_etf_symbols(master: pd.DataFrame, *, dataset: str, trade_date: str) -> list[str]:
    suffix = ".SH" if dataset == "etf_sh_cons" else ".SZ"
    frame = master.copy()
    if "list_status" in frame.columns:
        frame = frame[frame["list_status"].astype("string").isin(["L", "D"])]
    if "list_date" in frame.columns:
        listed = pd.to_datetime(frame["list_date"], errors="coerce")
        frame = frame[listed.isna() | (listed <= pd.Timestamp(trade_date))]
    return sorted(
        {
            str(value).strip().upper()
            for value in frame["ts_code"].dropna().tolist()
            if str(value).strip().upper().endswith(suffix)
        }
    )


def _historical_etf_active_ranges(
    master: pd.DataFrame, *, start: date, end: date
) -> dict[str, tuple[date, date]]:
    """Clip each ETF to the requested history window using the current master."""

    if master.empty:
        return {}
    if "ts_code" not in master.columns:
        raise RuntimeError("etf_basic did not provide ts_code for constituent planning")
    frame = master.copy()
    if "list_status" in frame.columns:
        frame = frame[frame["list_status"].astype("string").isin(["L", "D"])]
    ranges: dict[str, tuple[date, date]] = {}
    for _, row in frame.iterrows():
        raw_symbol = row.get("ts_code")
        if pd.isna(raw_symbol):
            continue
        symbol = str(raw_symbol).strip().upper()
        if not symbol.endswith((".SH", ".SZ")):
            continue
        listed_value = pd.to_datetime(row.get("list_date"), errors="coerce")
        delisted_value = pd.to_datetime(row.get("delist_date"), errors="coerce")
        listed_at = start if pd.isna(listed_value) else listed_value.date()
        delisted_at = end if pd.isna(delisted_value) else delisted_value.date()
        clipped = (max(start, listed_at), min(end, delisted_at))
        if clipped[1] < clipped[0]:
            continue
        previous = ranges.get(symbol)
        ranges[symbol] = (
            (min(previous[0], clipped[0]), max(previous[1], clipped[1])) if previous else clipped
        )
    return ranges


def _institutional_history_specs(
    context: Context,
    *,
    start: date,
    end: date,
    trading_dates: list[str],
    max_attempts: int,
) -> list[FetchSpec]:
    """Plan dense ETF baskets by symbol/range while retaining the bundle contract."""

    base_specs = supplemental_specs(
        "cn_institutional",
        start=start,
        end=end,
        trading_dates=trading_dates,
        max_attempts=max_attempts,
    )
    fund_specs = supplemental_specs(
        "cn_funds",
        start=end,
        end=end,
        trading_dates=[],
        max_attempts=max_attempts,
    )
    master_specs = [spec for spec in fund_specs if spec.dataset == "etf_basic"]
    _, master_rows, _ = _run_paginated_specs(
        context,
        "complete ETF master for constituent planning",
        master_specs,
    )
    master = context.storage.read_units(
        [row for row in master_rows if row["dataset"] == "etf_basic"]
    )
    active_ranges = _historical_etf_active_ranges(master, start=start, end=end)
    constituent_specs = etf_constituent_history_specs(
        active_ranges,
        max_attempts=max_attempts,
    )

    start_text = compact_date(start)
    end_text = compact_date(end)
    legacy_keys = []
    for row in context.checkpoint.unfinished_units({"etf_sh_cons", "etf_sz_cons"}):
        scope = dict(row.get("scope_json") or {})
        trade_date = str(scope.get("trade_date") or "")
        if (
            len(trade_date) == 8
            and start_text <= trade_date <= end_text
            and not scope.get("start_date")
        ):
            legacy_keys.append(str(row["unit_key"]))
    superseded = context.checkpoint.supersede_units(
        legacy_keys,
        "replaced by ETF symbol/date-range pagination",
    )
    if superseded:
        console.print(f"retired legacy per-ETF/per-day constituent units: {superseded}")

    return [
        spec for spec in base_specs if spec.dataset not in {"etf_sh_cons", "etf_sz_cons"}
    ] + constituent_specs


@app.command()
def probe() -> None:
    """Verify credentials and the Tushare-compatible response shape."""
    context = load_context()
    result = context.provider.fetch(
        "stock_basic",
        {"list_status": "L", "limit": 5},
        ("ts_code", "symbol", "name", "list_date"),
    )
    console.print(
        f"[green]provider OK[/green] api={context.settings.api_url} rows={len(result.rows)} "
        f"columns={','.join(result.columns)}"
    )


@app.command()
def bootstrap(
    profile: Annotated[str, typer.Option(help="core, research, or full")] = "full",
    start: Annotated[str, typer.Option(help="YYYY-MM-DD")] = "2016-01-01",
    snapshot_start: Annotated[
        str,
        typer.Option(
            "--snapshot-start",
            help="YYYY-MM-DD start of the merged BaoStock + primary-source snapshot",
        ),
    ] = "2008-01-01",
    end: Annotated[str, typer.Option(help="YYYY-MM-DD or latest")] = "latest",
    snapshot_name: Annotated[str | None, typer.Option("--snapshot-name")] = None,
    build_qlib: Annotated[
        bool, typer.Option("--build-qlib/--no-build-qlib", help="Build Qlib .bin after snapshot")
    ] = True,
    download_only: Annotated[
        bool,
        typer.Option("--download-only", help="Stop after durable download units complete"),
    ] = False,
    incremental: Annotated[
        bool,
        typer.Option(
            "--incremental",
            help="Internal scheduler mode for a bounded primary-source refresh",
            hidden=True,
        ),
    ] = False,
) -> None:
    """Plan, download, verify, and snapshot an initialization range."""
    if profile not in {"core", "research", "full"}:
        raise typer.BadParameter("profile must be core, research, or full")
    start_date = parse_date(start)
    snapshot_start_date = parse_date(snapshot_start)
    end_date = parse_date(end, latest=today_cn())
    if incremental is not True and start_date != date(2016, 1, 1):
        raise typer.BadParameter(
            "full bootstrap primary download start must equal 2016-01-01; "
            "use the incremental scheduler for later updates"
        )
    if incremental is True and download_only is not True:
        raise typer.BadParameter("incremental scheduler mode requires --download-only")
    if build_qlib and profile != "full":
        raise typer.BadParameter(
            "Qlib research finalization requires --profile full; "
            "core/research profiles are download-only subsets"
        )
    if end_date < start_date:
        raise typer.BadParameter("end must not be before start")
    if snapshot_start_date > start_date:
        raise typer.BadParameter("snapshot-start must not be after the primary download start")
    context = load_context()
    max_attempts = context.settings.max_request_attempts
    planned_reference = context.planner.plan_reference(start_date, end_date, max_attempts)
    console.print(f"planned reference units: +{planned_reference}")
    _run_phase(context, "stock and calendar reference", {"stock_basic", "trade_cal"})
    index_specs = context.planner.index_catalog_specs(max_attempts, as_of=end_date)
    _, _, index_inserted = _run_paginated_specs(
        context,
        "complete index catalog",
        index_specs,
    )
    console.print(
        f"planned complete index catalog: {len(index_specs)} initial, "
        f"+{index_inserted} inserted with pagination"
    )
    _require_selected_plan_complete(
        context,
        {"stock_basic", "trade_cal", "index_basic"},
        label="reference phase",
        snapshot_start=snapshot_start_date,
        snapshot_end=end_date,
    )

    planned = context.planner.plan_profile(profile, start_date, end_date, max_attempts)
    console.print(f"planned data units: {json.dumps(planned, ensure_ascii=False)}")
    daily_datasets = {definition.name for definition in CORE_DAILY}
    daily_datasets.update({"index_daily", "index_dailybasic", "index_weight"})
    _run_phase(context, "core market data", daily_datasets)
    if profile in {"research", "full"}:
        _run_phase(context, "research daily data", {item.name for item in RESEARCH_DAILY})
        _run_phase(
            context,
            "industry, disclosure, and ETF data",
            {
                "fund_basic",
                "fund_daily",
                "fund_adj",
                "index_classify",
                "disclosure_date",
            },
        )
        _require_selected_plan_complete(
            context,
            {"index_classify"},
            label="Shenwan classification phase",
            snapshot_start=snapshot_start_date,
            snapshot_end=end_date,
        )
        planned_members = context.planner.plan_industry_members(max_attempts, as_of=end_date)
        console.print(f"planned historical industry membership units: +{planned_members}")
        _run_phase(context, "historical industry members", {"index_member_all"})
        _require_selected_plan_complete(
            context,
            {"index_member_all", "index_weight"},
            label="benchmark industry residual prerequisites",
            snapshot_start=snapshot_start_date,
            snapshot_end=end_date,
        )
        planned_residual_members = (
            context.planner.plan_benchmark_industry_residual_members(
                snapshot_start_date,
                end_date,
                max_attempts,
                as_of=end_date,
            )
        )
        console.print(
            "planned benchmark residual industry membership units: "
            f"+{planned_residual_members}"
        )
        _run_phase(
            context,
            "benchmark residual industry members",
            {"index_member_all"},
        )
        _require_selected_plan_complete(
            context,
            {"index_member_all"},
            label="benchmark residual industry phase",
            snapshot_start=snapshot_start_date,
            snapshot_end=end_date,
        )
    if profile == "full":
        bulk_specs = a_share_bulk_history_specs(
            start=start_date,
            end=end_date,
            max_attempts=max_attempts,
        )
        bulk_specs, _, bulk_inserted = _run_paginated_specs(
            context,
            "full-market fundamentals and corporate events",
            bulk_specs,
        )
        console.print(
            f"planned full-market financial/event units: "
            f"{len(bulk_specs)} initial, +{bulk_inserted} inserted with pagination"
        )
        institutional_specs = _institutional_history_specs(
            context,
            start=start_date,
            end=end_date,
            trading_dates=context.planner.trading_dates(start_date, end_date),
            max_attempts=max_attempts,
        )
        institutional_specs, _, institutional_inserted = _run_paginated_specs(
            context,
            "institutional research and enhanced data",
            institutional_specs,
        )
        console.print(
            "planned institutional research units: "
            f"{len(institutional_specs)} initial, "
            f"+{institutional_inserted} inserted with pagination"
        )
        news_plan = context.planner.news_specs(start_date, end_date, max_attempts)
        _run_paginated_specs(context, "market news", news_plan)

    snapshot_datasets = _snapshot_datasets(
        profile,
        available=set(context.checkpoint.datasets()),
    )
    _require_selected_plan_complete(
        context,
        snapshot_datasets,
        label=f"{profile} bootstrap plan",
        snapshot_start=snapshot_start_date,
        snapshot_end=end_date,
        required_datasets=_required_profile_datasets(profile),
    )

    if download_only:
        console.print("[bold green]download phase complete[/bold green]")
        return

    report = verify_downloads(
        context.checkpoint,
        context.settings.data_root,
        snapshot_start=snapshot_start_date,
        snapshot_end=end_date,
        require_all_planned=True,
        dataset_filter=snapshot_datasets,
        required_datasets=_required_profile_datasets(profile),
        profile=profile,
    )
    report_path = context.settings.data_root / "verification" / "latest.json"
    write_report(report, report_path)
    if not report["ok"]:
        console.print(f"[red]verification failed[/red]: {report_path}")
        for error in report["errors"][:20]:
            console.print(f"  - {error}")
        raise typer.Exit(3)

    name = snapshot_name or (
        f"cn-{snapshot_start_date:%Y%m%d}-{end_date:%Y%m%d}-"
        f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    )
    snapshot_path = _build_snapshot(
        context,
        name,
        snapshot_start_date,
        end_date,
        profile,
        quality_gate=quality_gate_payload(report),
    )
    write_report(report, snapshot_path / "verification.json")
    if build_qlib:
        qlib_path = _build_qlib(context, snapshot_path, staging_only=False)
        console.print(f"[green]Qlib dataset built[/green]: {qlib_path}")
    console.print(f"[bold green]bootstrap complete[/bold green]: {snapshot_path}")


def _run_legacy_market_specs(
    context: Context,
    specs: list[FetchSpec],
    label: str,
) -> tuple[int, int]:
    keys = {spec.unit_key for spec in specs}
    datasets = {spec.dataset for spec in specs}
    api_names = {spec.api_name for spec in specs}
    inserted = context.checkpoint.add(specs)
    context.checkpoint.retry_failed_units(keys)
    summary = context.runner.run(datasets, api_names=api_names)
    rows = _require_specs_complete(context, specs)
    console.print(
        f"{label}: planned={len(specs)} inserted={inserted} "
        f"succeeded={summary.succeeded} "
        f"rows={sum(int(row.get('row_count') or 0) for row in rows)}"
    )
    return inserted, summary.succeeded


@app.command("bootstrap-legacy-market")
def bootstrap_legacy_market(
    start: Annotated[str, typer.Option(help="YYYY-MM-DD")] = "2008-01-01",
    end: Annotated[str, typer.Option(help="YYYY-MM-DD")] = "2015-12-31",
    validation_report: Annotated[
        Path | None,
        typer.Option(
            "--validation-report",
            help="Successful 2016 BaoStock/Tushare overlap report",
        ),
    ] = None,
    result_path: Annotated[Path | None, typer.Option("--result")] = None,
) -> None:
    """Backfill audited pre-2016 A-share market data from BaoStock.

    Only the calendar, unadjusted daily bars, BaoStock's published valuation
    fields, and a derived adjustment factor are imported.  Fundamentals, news,
    and other datasets are not fabricated for dates their sources cannot cover.
    """

    start_date = parse_date(start)
    end_date = parse_date(end)
    if end_date < start_date:
        raise typer.BadParameter("end must not be before start")
    if end_date >= date(2016, 1, 1):
        raise typer.BadParameter(
            "legacy production import must end before 2016-01-01; "
            "use the overlap validator for cross-source comparison"
        )
    if validation_report is None or not validation_report.is_file():
        raise typer.BadParameter(
            "a successful --validation-report is required before legacy import"
        )
    try:
        validation = json.loads(validation_report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise typer.BadParameter("validation report is unreadable") from exc
    if (
        not isinstance(validation, dict)
        or validation.get("ok") is not True
        or validation.get("source") != "baostock-0.9.3"
        or validation.get("reference_source") != PRIMARY_OVERLAP_PROVIDER
        or validation.get("policy_version") != BAOSTOCK_OVERLAP_POLICY_VERSION
        or str(validation.get("start_date") or "") > "2016-01-01"
        or str(validation.get("end_date") or "") < "2016-12-31"
    ):
        raise typer.BadParameter("validation report did not pass the required 2016 overlap gate")

    context = load_context(
        require_credentials=False,
        progress_path=result_path,
        progress_target={
            "kind": "legacy_market_backfill",
            "source": "baostock",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        },
    )
    try:
        require_current_primary_overlap_evidence(validation, context.checkpoint)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    reference_specs = baostock_reference_specs(
        start_date,
        end_date,
        max_attempts=context.settings.max_request_attempts,
    )
    with BaoStockProvider() as provider:
        context.provider = provider
        context.runner = DownloadRunner(
            checkpoint=context.checkpoint,
            storage=context.storage,
            provider=provider,
            # BaoStock's Python client owns one process-global socket.
            workers=1,
        )
        context.report_progress(
            "prerequisites",
            "BaoStock calendar and historical stock master",
            {"trade_cal", "baostock_stock_basic"},
            force=True,
        )
        _run_legacy_market_specs(
            context,
            reference_specs,
            "BaoStock calendar and historical stock master",
        )
        codes = planned_baostock_universe(
            context.checkpoint,
            context.storage,
            start=start_date,
            end=end_date,
        )
        if not codes:
            raise RuntimeError("BaoStock historical A-share universe is empty")
        history_specs = baostock_history_specs(
            codes,
            start_date,
            end_date,
            max_attempts=context.settings.max_request_attempts,
        )
        context.report_progress(
            "downloading",
            "BaoStock pre-2016 market history",
            LEGACY_MARKET_DATASETS,
            force=True,
        )
        inserted, succeeded = _run_legacy_market_specs(
            context,
            history_specs,
            "BaoStock pre-2016 market history",
        )

    result = {
        "status": "succeeded",
        "source": "baostock",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "symbols": len(codes),
        "planned_units": len(history_specs),
        "inserted_units": inserted,
        "succeeded_this_run": succeeded,
    }
    _write_optional_result(result_path, result)
    console.print_json(json.dumps(result, ensure_ascii=False))


@app.command("validate-baostock-overlap")
def validate_baostock_overlap_command(
    start: Annotated[str, typer.Option(help="YYYY-MM-DD")] = "2016-01-01",
    end: Annotated[str, typer.Option(help="YYYY-MM-DD")] = "2016-12-31",
    symbols: Annotated[
        str,
        typer.Option(help="Comma-separated Tushare codes; empty uses the audited sample"),
    ] = "",
    result_path: Annotated[Path | None, typer.Option("--result")] = None,
) -> None:
    """Gate legacy imports by comparing BaoStock with the primary source."""

    start_date = parse_date(start)
    end_date = parse_date(end)
    if end_date < start_date:
        raise typer.BadParameter("end must not be before start")
    try:
        selected_symbols = require_audited_overlap_symbols(
            tuple(_split_codes(symbols)) or DEFAULT_OVERLAP_SYMBOLS
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    context = load_context(require_credentials=False)
    with BaoStockProvider() as provider:
        report = validate_baostock_overlap(
            context.checkpoint,
            context.storage,
            provider,
            start=start_date,
            end=end_date,
            symbols=selected_symbols,
        )
    _write_optional_result(result_path, report)
    console.print_json(json.dumps(report, ensure_ascii=False))
    if not report["ok"]:
        raise typer.Exit(3)


@app.command()
def status() -> None:
    """Show checkpoint counts and recent failures without requiring credentials."""
    context = load_context(require_credentials=False)
    table = Table("dataset", "status", "units", "rows")
    for row in context.checkpoint.counts():
        table.add_row(row["dataset"], row["status"], str(row["units"]), str(row["rows"]))
    console.print(table)
    failures = context.checkpoint.failures()
    if failures:
        console.print("[red]recent failures[/red]")
        for row in failures:
            console.print(f"{row['dataset']} {row['scope_json']}: {row['last_error']}")


@app.command("retry-failed")
def retry_failed() -> None:
    """Reset failed units and execute them again."""
    context = load_context()
    count = context.checkpoint.retry_failed()
    console.print(f"reset failed units: {count}")
    _run_phase(context, "retry", set(context.checkpoint.datasets()))


def _trigger_safe_mode_on_quality_gate_failure(settings: Any, report: dict[str, Any]) -> None:
    """Design 11.3: a failed data quality gate is a severe data anomaly.

    Never masks the original verification failure when the control plane is
    unreachable.
    """

    try:
        from quant_platform.safe_mode import SafeModeStore

        SafeModeStore(settings.database_url).activate(
            reason=f"数据质量门校验失败（{len(report['errors'])} 项错误）",
            source="data_quality_gate",
            actor="system",
            details={"errors": [str(item) for item in report["errors"]][:20]},
        )
    except Exception:  # noqa: BLE001 - the verification failure takes priority
        console.print("[red]safe_mode trigger failed; verification error stands[/red]")


@app.command()
def verify(
    snapshot_start: Annotated[
        str, typer.Option("--snapshot-start", help="Successor snapshot start date")
    ] = "2008-01-01",
    snapshot_end: Annotated[
        str, typer.Option("--snapshot-end", help="Successor snapshot end date")
    ] = "latest",
    allow_incomplete_plans: Annotated[
        bool,
        typer.Option(
            "--allow-incomplete-plans",
            help=(
                "Deprecated compatibility flag; production verification always "
                "requires the complete active plan"
            ),
        ),
    ] = False,
    profile: Annotated[
        str | None,
        typer.Option(
            "--profile",
            help=(
                "Strictly verify only the datasets that the selected snapshot "
                "profile will publish: core, research, full, research-assets, or "
                "global-reference"
            ),
        ),
    ] = None,
) -> None:
    """Validate checkpoints, files, checksums, empties, and duplicate core keys."""
    context = load_context(require_credentials=False)
    if allow_incomplete_plans:
        console.print(
            "[yellow]--allow-incomplete-plans is deprecated and ignored; "
            "production verification remains strict[/yellow]"
        )
    if profile is not None and profile not in SNAPSHOT_PROFILES:
        raise typer.BadParameter(
            "profile must be core, research, full, research-assets, or "
            "global-reference"
        )
    dataset_filter = (
        _snapshot_datasets(profile, available=set(context.checkpoint.datasets()))
        if profile is not None
        else None
    )
    report = verify_downloads(
        context.checkpoint,
        context.settings.data_root,
        snapshot_start=parse_date(snapshot_start),
        snapshot_end=parse_date(snapshot_end, latest=today_cn()),
        require_all_planned=True,
        dataset_filter=dataset_filter,
        required_datasets=(
            _required_profile_datasets(profile) if profile is not None else None
        ),
        profile=profile,
    )
    path = context.settings.data_root / "verification" / "latest.json"
    write_report(report, path)
    console.print_json(json.dumps(report, ensure_ascii=False))
    if not report["ok"]:
        _trigger_safe_mode_on_quality_gate_failure(context.settings, report)
        raise typer.Exit(3)


@app.command()
def snapshot(
    name: Annotated[str | None, typer.Option()] = None,
    start: Annotated[str, typer.Option()] = "2008-01-01",
    end: Annotated[str, typer.Option()] = "latest",
    profile: Annotated[
        str,
        typer.Option(
            help=(
                "core, research, full, the isolated research-assets "
                "profile (trade_cal + research_report only), or the isolated "
                "global-reference profile (peripheral markets only)"
            )
        ),
    ] = "core",
    industry_history_anchor: Annotated[
        str | None,
        typer.Option(
            "--industry-history-anchor",
            help=(
                "One verified older snapshot used once to restore missing PIT industry "
                "rows for explicitly delisted stocks in a new lineage root"
            ),
        ),
    ] = None,
) -> None:
    """Build an immutable compacted Parquet snapshot from successful units."""
    context = load_context(require_credentials=False)
    start_date = parse_date(start)
    end_date = parse_date(end, latest=today_cn())
    if name is None:
        prefix = (
            "global" if profile == GLOBAL_REFERENCE_SNAPSHOT_PROFILE else "cn"
        )
        name = f"{prefix}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    # Verify before building so every snapshot manifest records an explicit
    # quality gate; Qlib builds refuse snapshots without quality_gate.ok=true.
    selected_datasets = _snapshot_datasets(
        profile,
        available=set(context.checkpoint.datasets()),
    )
    report = verify_downloads(
        context.checkpoint,
        context.settings.data_root,
        snapshot_start=start_date,
        snapshot_end=end_date,
        require_all_planned=True,
        dataset_filter=selected_datasets,
        required_datasets=_required_profile_datasets(profile),
        profile=profile,
    )
    write_report(report, context.settings.data_root / "verification" / "latest.json")
    if not report["ok"]:
        console.print("[red]verification failed; snapshot build refused[/red]")
        for error in report["errors"][:20]:
            console.print(f"  - {error}")
        _trigger_safe_mode_on_quality_gate_failure(context.settings, report)
        raise typer.Exit(3)
    path = _build_snapshot(
        context,
        name,
        start_date,
        end_date,
        profile,
        quality_gate=quality_gate_payload(report),
        industry_history_anchor=industry_history_anchor,
    )
    write_report(report, path / "verification.json")
    console.print(path)


@app.command("snapshot-ingested-at-successor")
def snapshot_ingested_at_successor(
    source: Annotated[
        str,
        typer.Option(
            "--source",
            help="Immutable source snapshot whose exact work units are re-verified",
        ),
    ],
    name: Annotated[
        str,
        typer.Option(
            "--name",
            help="New immutable successor name; the source is never overwritten",
        ),
    ],
    dataset: Annotated[
        str,
        typer.Option(
            "--dataset",
            help="Only this affected dataset is re-verified and rebuilt",
        ),
    ] = "fina_indicator",
) -> None:
    """Recover missing row acquisition times from the succeeded-unit ledger."""

    if dataset != "fina_indicator":
        raise typer.BadParameter(
            "--dataset currently permits only the audited fina_indicator recovery"
        )
    context = load_context(require_credentials=False)
    path = context.storage.build_ingested_at_successor(
        name=name,
        source_snapshot=context.storage.snapshots_root / source,
        checkpoint=context.checkpoint,
        datasets={dataset},
    )
    console.print(path)


@app.command("margin-eligibility")
def margin_eligibility(
    start: Annotated[str, typer.Option(help="YYYY-MM-DD")] = "2024-01-01",
    end: Annotated[str, typer.Option(help="YYYY-MM-DD or latest")] = "latest",
    result_path: Annotated[Path | None, typer.Option("--result")] = None,
) -> None:
    """Download daily full-market margin-eligible security evidence."""
    start_date = parse_date(start)
    end_date = parse_date(end, latest=today_cn())
    if end_date < start_date:
        raise typer.BadParameter("end must not be before start")
    context = load_context(
        progress_path=result_path,
        progress_target={
            "kind": "margin_eligibility_download",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        },
    )
    context.report_progress("planning", "margin eligibility planning", {MARGIN_DATASET}, force=True)
    calendar = FetchSpec(
        dataset="trade_cal",
        api_name="trade_cal",
        scope={
            "exchange": "SSE",
            "start": compact_date(start_date),
            "end": compact_date(end_date),
        },
        params={
            "exchange": "SSE",
            "start_date": compact_date(start_date),
            "end_date": compact_date(end_date),
        },
        fields=REFERENCE_FIELDS["trade_cal"],
        max_attempts=context.settings.max_request_attempts,
    )
    context.checkpoint.add([calendar])
    context.checkpoint.retry_failed_units([calendar.unit_key])
    _run_phase(context, "trading calendar", {"trade_cal"})
    _require_specs_complete(context, [calendar])
    trading_dates = context.planner.trading_dates(start_date, end_date)
    specs = context.execution_planner.plan_margin(
        trading_dates,
        context.settings.max_request_attempts,
    )
    _run_phase(context, "margin eligibility", {MARGIN_DATASET})
    rows = _require_specs_complete(context, specs)
    result = {
        "status": "succeeded",
        "dataset": MARGIN_DATASET,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "trading_days": len(trading_dates),
        "units": len(rows),
        "rows": sum(int(row.get("row_count") or 0) for row in rows),
    }
    _write_optional_result(result_path, result)
    console.print_json(json.dumps(result, ensure_ascii=False))


@app.command("core-intraday")
def core_intraday(
    start: Annotated[str, typer.Option(help="YYYY-MM-DD")] = "2024-01-01",
    end: Annotated[str, typer.Option(help="YYYY-MM-DD or latest")] = "latest",
    etfs: Annotated[str, typer.Option(help="Comma-separated Tushare/Qlib ETF codes")] = "",
    stocks: Annotated[str, typer.Option(help="Comma-separated Tushare/Qlib stock codes")] = "",
    indices: Annotated[str, typer.Option(help="Comma-separated index codes")] = "",
    futures: Annotated[str, typer.Option(help="Comma-separated futures contract codes")] = "",
    options: Annotated[str, typer.Option(help="Comma-separated option contract codes")] = "",
    auto_universe: Annotated[
        bool,
        typer.Option(
            "--auto-universe/--manual-universe",
            help="Select the core universe from downloaded masters",
        ),
    ] = False,
    max_stocks: Annotated[int, typer.Option(min=0, max=500)] = 100,
    max_options: Annotated[int, typer.Option(min=0, max=500)] = 100,
    etf_categories: Annotated[
        str, typer.Option(help="Comma-separated ETF groups: broad,industry,gold,bond")
    ] = "broad,industry,gold,bond",
    snapshot_name: Annotated[str | None, typer.Option("--snapshot-name")] = None,
    source_lineage_id: Annotated[
        str | None,
        typer.Option(
            "--source-lineage-id",
            help="Verified daily-source lineage paired with this execution snapshot",
        ),
    ] = None,
    daily_source_dataset: Annotated[
        str | None,
        typer.Option(
            "--daily-source-dataset",
            help="Exact verified daily Qlib dataset selected by the controller",
        ),
    ] = None,
    result_path: Annotated[Path | None, typer.Option("--result")] = None,
) -> None:
    """Download bounded 1-minute windows and build pair-execution evidence."""
    start_date = parse_date(start)
    end_date = parse_date(end, latest=today_cn())
    if end_date < start_date:
        raise typer.BadParameter("end must not be before start")
    if not source_lineage_id or len(source_lineage_id) != 64 or any(
        character not in "0123456789abcdef" for character in source_lineage_id.lower()
    ):
        raise typer.BadParameter(
            "core intraday download requires a verified daily --source-lineage-id"
        )
    symbols_by_dataset = {
        dataset: values
        for dataset, values in {
            "etf_1m": _split_codes(etfs),
            "liquid_stocks_1m": _split_codes(stocks),
            "indices_1m": _split_codes(indices),
            "futures_1m": _split_codes(futures),
            "options_1m": _split_codes(options),
        }.items()
        if values
    }
    context = load_context(
        progress_path=result_path,
        progress_target={
            "kind": "core_intraday_download",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        },
    )
    source_lineage_evidence = _require_local_daily_source_lineage(
        context,
        source_lineage_id=source_lineage_id,
        daily_source_dataset=daily_source_dataset,
        start_date=start_date,
        end_date=end_date,
    )
    price_limit_rows = _source_snapshot_dataset_rows(
        context,
        source_lineage_evidence=source_lineage_evidence,
        dataset="stk_limit",
        start_date=start_date,
        end_date=end_date,
    )
    trading_dates = _source_daily_trading_dates(
        context,
        source_lineage_evidence=source_lineage_evidence,
        start_date=start_date,
        end_date=end_date,
    )
    universe_evidence: dict | None = None
    if auto_universe:
        universe_rows: dict[str, list[dict[str, Any]]] = {}
        for dataset in ("daily", "stock_basic", "fut_mapping", "opt_daily"):
            try:
                universe_rows[dataset] = _source_snapshot_dataset_rows(
                    context,
                    source_lineage_evidence=source_lineage_evidence,
                    dataset=dataset,
                    start_date=start_date,
                    end_date=end_date,
                )
            except ValueError:
                if dataset in {"daily", "stock_basic"}:
                    raise
                universe_rows[dataset] = []
        selected = select_intraday_universe(
            {
                dataset: context.storage.read_units(rows)
                for dataset, rows in universe_rows.items()
            },
            max_stocks=max_stocks,
            max_options=max_options,
            etf_categories=tuple(_split_codes(etf_categories)),
            start=start_date,
            end=end_date,
        )
        for dataset, values in selected.symbols_by_dataset.items():
            symbols_by_dataset[dataset] = sorted(
                set(symbols_by_dataset.get(dataset, [])) | set(values)
            )
        universe_source_units = [
            {
                "dataset": dataset,
                "unit_key": str(row["unit_key"]),
                "sha256": str(row["sha256"]),
                "row_count": int(row.get("row_count") or 0),
            }
            for dataset, rows in sorted(universe_rows.items())
            for row in sorted(rows, key=lambda item: str(item["unit_key"]))
        ]
        universe_evidence = {
            **selected.evidence,
            "source_snapshot": source_lineage_evidence["source_snapshot"],
            "source_lineage_id": source_lineage_id.lower(),
            "source_units_sha256": hashlib.sha256(
                json.dumps(
                    universe_source_units,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "source_units": universe_source_units,
        }
    if not symbols_by_dataset:
        raise typer.BadParameter(
            "at least one ETF, stock, index, future, or option code is required"
        )
    context.progress.set_target(
        symbols=sum(len(values) for values in symbols_by_dataset.values()),
        frequency="1min",
    )
    context.report_progress(
        "planning", "core intraday planning", set(symbols_by_dataset), force=True
    )
    required_margin_specs = margin_specs(
        trading_dates,
        max_attempts=context.settings.max_request_attempts,
    )
    margin_rows = _require_specs_complete(
        context,
        required_margin_specs,
        hint="run margin-eligibility for the same date range first",
    )
    specs = context.execution_planner.plan_minutes(
        symbols_by_dataset,
        start_date,
        end_date,
        context.settings.max_request_attempts,
        trading_dates=trading_dates,
    )
    minute_datasets = set(symbols_by_dataset)
    specs, minute_rows, _ = _run_paginated_specs(context, "core intraday", specs)
    _require_symbol_coverage(specs, minute_rows)
    normalized_symbols = {
        dataset: sorted({str(spec.params["ts_code"]) for spec in specs if spec.dataset == dataset})
        for dataset in sorted(minute_datasets)
    }

    name = snapshot_name or (
        f"execution-{start_date:%Y%m%d}-{end_date:%Y%m%d}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    )
    selected: dict[str, list[dict]] = {
        MARGIN_DATASET: margin_rows,
        "stk_limit": price_limit_rows,
    }
    for dataset in sorted(minute_datasets):
        keys = {spec.unit_key for spec in specs if spec.dataset == dataset}
        selected[dataset] = [row for row in minute_rows if row["unit_key"] in keys]
    quality_gate = _explicit_execution_quality_gate(
        context,
        selected=selected,
        start_date=start_date,
        end_date=end_date,
        profile="pair_execution",
    )
    context.report_progress(
        "snapshot", "building immutable execution snapshot", minute_datasets, force=True
    )
    snapshot_path = _build_execution_snapshot(
        context,
        name=name,
        selected=selected,
        start_date=start_date,
        end_date=end_date,
        symbols_by_dataset=normalized_symbols,
        universe_evidence=universe_evidence,
        source_lineage_id=source_lineage_id.lower(),
        source_lineage_evidence=source_lineage_evidence,
        quality_gate=quality_gate,
    )
    result = {
        "status": "succeeded",
        "snapshot_name": name,
        "snapshot_path": str(snapshot_path),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "datasets": {
            dataset: {
                "units": len(rows),
                "rows": sum(int(row.get("row_count") or 0) for row in rows),
            }
            for dataset, rows in selected.items()
        },
        "universe": universe_evidence or {"mode": "manual"},
    }
    _write_optional_result(result_path, result)
    console.print_json(json.dumps(result, ensure_ascii=False))


@app.command("ashare-5m")
def ashare_5m(
    start: Annotated[str, typer.Option(help="YYYY-MM-DD")] = "2024-01-01",
    end: Annotated[str, typer.Option(help="YYYY-MM-DD or latest")] = "latest",
    snapshot_name: Annotated[str | None, typer.Option("--snapshot-name")] = None,
    source_lineage_id: Annotated[
        str | None,
        typer.Option(
            "--source-lineage-id",
            help="Verified daily-source lineage paired with this execution snapshot",
        ),
    ] = None,
    daily_source_dataset: Annotated[
        str | None,
        typer.Option(
            "--daily-source-dataset",
            help="Exact verified daily Qlib dataset selected by the controller",
        ),
    ] = None,
    result_path: Annotated[Path | None, typer.Option("--result")] = None,
) -> None:
    """Download resumable 5-minute bars for every A-share active in the range."""

    start_date = parse_date(start)
    end_date = parse_date(end, latest=today_cn())
    if end_date < start_date:
        raise typer.BadParameter("end must not be before start")
    if not source_lineage_id or len(source_lineage_id) != 64 or any(
        character not in "0123456789abcdef" for character in source_lineage_id.lower()
    ):
        raise typer.BadParameter(
            "A-share five-minute download requires a verified daily --source-lineage-id"
        )
    context = load_context(
        progress_path=result_path,
        progress_target={
            "kind": "ashare_5m_download",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "frequency": "5min",
        },
    )
    source_lineage_evidence = _require_local_daily_source_lineage(
        context,
        source_lineage_id=source_lineage_id,
        daily_source_dataset=daily_source_dataset,
        start_date=start_date,
        end_date=end_date,
    )
    price_limit_rows = _source_snapshot_dataset_rows(
        context,
        source_lineage_evidence=source_lineage_evidence,
        dataset="stk_limit",
        start_date=start_date,
        end_date=end_date,
    )
    stock_master_rows = _source_snapshot_dataset_rows(
        context,
        source_lineage_evidence=source_lineage_evidence,
        dataset="stock_basic",
        start_date=start_date,
        end_date=end_date,
    )
    trading_dates = _source_daily_trading_dates(
        context,
        source_lineage_evidence=source_lineage_evidence,
        start_date=start_date,
        end_date=end_date,
    )
    master = context.storage.read_units(stock_master_rows)
    active_ranges = _historical_a_share_active_ranges(master, start=start_date, end=end_date)
    symbols = sorted(active_ranges)

    if not symbols:
        raise RuntimeError("stock_basic produced an empty historical A-share universe")
    context.progress.set_target(symbols=len(symbols))
    context.report_progress("planning", "full A-share 5-minute planning", {"ashare_5m"}, force=True)
    specs = context.execution_planner.plan_minutes(
        {"ashare_5m": symbols},
        start_date,
        end_date,
        context.settings.max_request_attempts,
        freq="5min",
        active_ranges_by_dataset={"ashare_5m": active_ranges},
        trading_dates=trading_dates,
    )
    specs, rows, _ = _run_paginated_specs(context, "full A-share 5-minute bars", specs)
    _require_symbol_coverage(specs, rows)
    name = snapshot_name or (
        f"ashare-5m-{start_date:%Y%m%d}-{end_date:%Y%m%d}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    )
    execution_selected = {
        "ashare_5m": rows,
        "stk_limit": price_limit_rows,
    }
    quality_gate = _explicit_execution_quality_gate(
        context,
        selected=execution_selected,
        start_date=start_date,
        end_date=end_date,
        profile="ashare_intraday",
    )
    quality_gate["daily_source_evidence_sha256"] = source_lineage_evidence[
        "evidence_sha256"
    ]
    context.report_progress(
        "snapshot", "building immutable 5-minute snapshot", {"ashare_5m"}, force=True
    )
    snapshot_path = _build_execution_snapshot(
        context,
        name=name,
        selected=execution_selected,
        start_date=start_date,
        end_date=end_date,
        symbols_by_dataset={"ashare_5m": symbols},
        universe_evidence={
            "mode": "historically_active_a_share_master",
            "source": "stock_basic",
            "count": len(symbols),
        },
        frequency="5min",
        profile="ashare_intraday",
        source_lineage_id=source_lineage_id.lower(),
        source_lineage_evidence=source_lineage_evidence,
        quality_gate=quality_gate,
    )
    write_report(quality_gate, snapshot_path / "verification.json")
    result = {
        "status": "succeeded",
        "dataset": "ashare_5m",
        "snapshot_name": name,
        "snapshot_path": str(snapshot_path),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "symbols": len(symbols),
        "units": len(rows),
        "rows": sum(int(row.get("row_count") or 0) for row in rows),
        "frequency": "5min",
    }
    _write_optional_result(result_path, result)
    console.print_json(json.dumps(result, ensure_ascii=False))


@app.command("cninfo-announcements")
def cninfo_announcements_command(
    ts_code: Annotated[str, typer.Option(help="Comma-separated Tushare codes to include")] = "",
    start: Annotated[str, typer.Option(help="YYYY-MM-DD announcement date")] = "2024-01-01",
    end: Annotated[str, typer.Option(help="YYYY-MM-DD or latest")] = "latest",
    limit: Annotated[
        int, typer.Option(min=0, help="Maximum announcements to download (0 = all)")
    ] = 0,
    regulatory_only: Annotated[
        bool,
        typer.Option(
            "--regulatory-only/--all-announcements",
            help="Download only regulatory letters identified by the governed title filter",
        ),
    ] = False,
    result_path: Annotated[Path | None, typer.Option("--result")] = None,
) -> None:
    """Download cninfo announcement PDFs discovered through the anns_d index."""
    start_date = parse_date(start)
    end_date = parse_date(end, latest=today_cn())
    if end_date < start_date:
        raise typer.BadParameter("end must not be before start")
    context = load_context(
        require_credentials=False,
        progress_path=result_path,
        progress_target={
            "kind": "cninfo_announcements_download",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        },
    )
    context.report_progress(
        "downloading", "cninfo announcement bodies", {"cninfo_announcements"}, force=True
    )

    def report_announcement_progress(progress: dict[str, int]) -> None:
        context.progress.set_target(**progress)
        context.report_progress(
            "downloading",
            "cninfo announcement bodies",
            {"cninfo_announcements"},
            force=True,
        )

    summary = download_cninfo_announcements(
        context.settings.data_root,
        ts_codes=set(_split_codes(ts_code)) or None,
        start=start_date,
        end=end_date,
        limit=limit or None,
        regulatory_only=regulatory_only,
        rate_gate=context.rate_gate,
        timeout_seconds=context.settings.timeout_seconds,
        max_attempts=context.settings.max_request_attempts,
        cooldown_seconds=context.settings.cooldown_seconds,
        progress_callback=report_announcement_progress,
    )
    context.report_progress(
        "verifying", "cninfo announcement quality audit", {"cninfo_announcements"}, force=True
    )
    quality_report = audit_cninfo_announcements(
        context.settings.data_root,
        ts_codes=set(_split_codes(ts_code)) or None,
        start=start_date,
        end=end_date,
        limit=limit or None,
        regulatory_only=regulatory_only,
        verify_hashes=True,
    )
    result = {
        "dataset": "cninfo_announcements",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "ts_codes": _split_codes(ts_code),
        "regulatory_only": regulatory_only,
        **summary.as_dict(),
        "quality_gate": {
            "ok": quality_report["ok"],
            "errors": quality_report["errors"],
            "warnings": quality_report["warnings"],
            "verified_at": quality_report["generated_at"],
        },
        "quality_report_path": quality_report["report_path"],
        "quality_report_sha256": quality_report["report_sha256"],
    }
    _write_optional_result(result_path, result)
    console.print_json(json.dumps(result, ensure_ascii=False))
    if summary.failed:
        raise typer.Exit(3)
    if not quality_report["ok"]:
        raise typer.Exit(4)


@app.command("announcement-nlp")
def announcement_nlp_command(
    ts_code: Annotated[str, typer.Option(help="Comma-separated Tushare codes to include")] = "",
    start: Annotated[str, typer.Option(help="YYYY-MM-DD announcement date")] = "2024-01-01",
    end: Annotated[str, typer.Option(help="YYYY-MM-DD or latest")] = "latest",
    category: Annotated[
        str,
        typer.Option(help="Comma-separated announcement|regulatory_letter; empty = all"),
    ] = "",
    limit: Annotated[
        int, typer.Option(min=0, help="Maximum announcements to process (0 = all)")
    ] = 0,
    batch_size: Annotated[
        int,
        typer.Option(
            min=1,
            max=ANNOUNCEMENT_MAX_BATCH_SIZE,
            help="Announcements per LLM request (strict item-id matching)",
        ),
    ] = ANNOUNCEMENT_DEFAULT_BATCH_SIZE,
    workers: Annotated[
        int,
        typer.Option(
            min=1,
            max=ANNOUNCEMENT_MAX_WORKERS,
            help="Concurrent bounded LLM requests",
        ),
    ] = ANNOUNCEMENT_DEFAULT_WORKERS,
    result_path: Annotated[Path | None, typer.Option("--result")] = None,
) -> None:
    """Extract structured NLP signal fields from downloaded announcement PDFs."""
    start_date = parse_date(start)
    end_date = parse_date(end, latest=today_cn())
    if end_date < start_date:
        raise typer.BadParameter("end must not be before start")
    categories = {part.strip() for part in category.split(",") if part.strip()}
    unknown = categories - {"announcement", "regulatory_letter"}
    if unknown:
        raise typer.BadParameter(f"unknown category: {sorted(unknown)}")
    context = load_context(
        require_credentials=False,
        progress_path=result_path,
        progress_target={
            "kind": "announcement_nlp",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        },
    )
    context.report_progress(
        "processing", "announcement NLP extraction", {"announcement_nlp"}, force=True
    )

    def report_announcement_nlp_progress(progress: dict[str, int]) -> None:
        context.progress.set_target(**progress)
        context.report_progress(
            "processing",
            "announcement NLP extraction",
            {"announcement_nlp"},
            force=True,
        )

    settings = context.settings
    secret_store = RuntimeSecretStore(settings.database_url, settings.platform_secret_key)
    summary = _produce_factors(
        "announcement-nlp",
        lambda: process_announcements(
            settings.data_root,
            ts_codes=set(_split_codes(ts_code)) or None,
            start=start_date,
            end=end_date,
            categories=categories or None,
            limit=limit or None,
            batch_size=batch_size,
            workers=workers,
            secret_store=secret_store,
            progress_callback=report_announcement_nlp_progress,
        ),
    )
    result = {
        "dataset": "announcement_nlp",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "ts_codes": _split_codes(ts_code),
        "category": sorted(categories),
        **summary.as_dict(),
    }
    _write_optional_result(result_path, result)
    console.print_json(json.dumps(result, ensure_ascii=False))
    if summary.failed:
        raise typer.Exit(3)


@app.command("corpus-nlp")
def corpus_nlp_command(
    dataset: Annotated[
        str,
        typer.Option(
            help=(
                "Comma-separated major_news,npr,cctv_news,irm_qa_sh,irm_qa_sz; "
                "empty = audited production sources (npr excluded until available)"
            )
        ),
    ] = "",
    ts_code: Annotated[str, typer.Option(help="Comma-separated Tushare codes to include")] = "",
    start: Annotated[str, typer.Option(help="YYYY-MM-DD publication date")] = "2024-01-01",
    end: Annotated[str, typer.Option(help="YYYY-MM-DD or latest")] = "latest",
    limit: Annotated[
        int, typer.Option(min=0, help="Maximum corpus items to process (0 = all)")
    ] = 0,
    batch_size: Annotated[
        int,
        typer.Option(
            min=1,
            max=100,
            help="Corpus items per LLM request (strict item-id matching)",
        ),
    ] = DEFAULT_BATCH_SIZE,
    workers: Annotated[
        int,
        typer.Option(
            min=1,
            max=MAX_WORKERS,
            help="Concurrent LLM requests under the shared global rate gate",
        ),
    ] = DEFAULT_WORKERS,
    major_news_per_day: Annotated[
        int,
        typer.Option(
            min=0,
            help="Deterministic major_news sample per publication day (0 = all)",
        ),
    ] = DEFAULT_MAJOR_NEWS_PER_DAY,
    irm_per_instrument_day: Annotated[
        int,
        typer.Option(
            min=0,
            help="Deterministic IR Q&A sample per instrument/day (0 = all)",
        ),
    ] = DEFAULT_IRM_PER_INSTRUMENT_DAY,
    result_path: Annotated[Path | None, typer.Option("--result")] = None,
) -> None:
    """Extract structured NLP signal fields from downloaded Tushare text corpora."""
    start_date = parse_date(start)
    end_date = parse_date(end, latest=today_cn())
    if end_date < start_date:
        raise typer.BadParameter("end must not be before start")
    datasets = {part.strip() for part in dataset.split(",") if part.strip()}
    unknown = datasets - set(SUPPORTED_CORPUS_DATASETS)
    if unknown:
        raise typer.BadParameter(f"unknown dataset: {sorted(unknown)}")
    context = load_context(
        require_credentials=False,
        progress_path=result_path,
        progress_target={
            "kind": "corpus_nlp",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        },
    )
    context.report_progress("processing", "corpus NLP extraction", {"corpus_nlp"}, force=True)

    def report_corpus_nlp_progress(progress: dict[str, int]) -> None:
        context.progress.set_target(**progress)
        context.report_progress("processing", "corpus NLP extraction", {"corpus_nlp"}, force=True)

    settings = context.settings
    secret_store = RuntimeSecretStore(settings.database_url, settings.platform_secret_key)
    summary = _produce_factors(
        "corpus-nlp",
        lambda: process_corpus(
            settings.data_root,
            datasets=datasets or None,
            ts_codes=set(_split_codes(ts_code)) or None,
            start=start_date,
            end=end_date,
            limit=limit or None,
            batch_size=batch_size,
            workers=workers,
            max_major_news_per_day=major_news_per_day or None,
            max_irm_per_instrument_day=irm_per_instrument_day or None,
            secret_store=secret_store,
            progress_callback=report_corpus_nlp_progress,
        ),
    )
    result = {
        "dataset": "corpus_nlp",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "datasets": sorted(datasets) or list(DEFAULT_CORPUS_DATASETS),
        "ts_codes": _split_codes(ts_code),
        **summary.as_dict(),
    }
    _write_optional_result(result_path, result)
    console.print_json(json.dumps(result, ensure_ascii=False))
    if summary.failed:
        raise typer.Exit(3)


@app.command("event-market-response")
def event_market_response_command(
    snapshot_name: Annotated[str, typer.Option(help="Verified immutable snapshot name")],
    horizons: Annotated[
        str, typer.Option(help="Comma-separated positive trading-session horizons")
    ] = ",".join(str(value) for value in DEFAULT_HORIZONS),
    benchmark_code: Annotated[
        str, typer.Option(help="Benchmark index ts_code")
    ] = DEFAULT_BENCHMARK,
    result_path: Annotated[Path | None, typer.Option("--result")] = None,
) -> None:
    """Build post-event training labels; never publishes them as live factors."""

    try:
        horizon_values = tuple(
            sorted({int(part.strip()) for part in horizons.split(",") if part.strip()})
        )
    except ValueError as exc:
        raise typer.BadParameter("horizons must be comma-separated integers") from exc
    if not horizon_values or any(value <= 0 for value in horizon_values):
        raise typer.BadParameter("horizons must contain positive integers")
    context = load_context(
        require_credentials=False,
        progress_path=result_path,
        progress_target={"kind": "event_market_response", "snapshot_name": snapshot_name},
    )
    context.report_progress(
        "processing", "post-event market-response labels", {"event_market_response"}, force=True
    )
    summary = _produce_factors(
        "event-market-response",
        lambda: process_event_market_response(
            context.settings.data_root,
            snapshot_name=snapshot_name,
            horizons=horizon_values,
            benchmark_code=benchmark_code,
        ),
    )
    result = {
        "dataset": "event_market_response_labels",
        "snapshot_name": snapshot_name,
        "horizons": list(horizon_values),
        "benchmark_code": benchmark_code,
        **summary.as_dict(),
    }
    _write_optional_result(result_path, result)
    console.print_json(json.dumps(result, ensure_ascii=False))


def _read_admin_asset_metadata(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    if path.is_symlink() or not path.is_file():
        raise typer.BadParameter("metadata JSON must be one regular, non-symlink file")
    if path.stat().st_size > 1024 * 1024:
        raise typer.BadParameter("metadata JSON must not exceed 1 MiB")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(f"metadata JSON is invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise typer.BadParameter("metadata JSON must contain one object")
    return {str(key): value for key, value in payload.items()}


def _parse_asset_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    normalized = value.strip()
    if normalized.endswith(("Z", "z")):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise typer.BadParameter("published-at must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise typer.BadParameter("published-at must include a timezone offset")
    return parsed


@app.command("research-asset-fetch-pdf")
def research_asset_fetch_pdf_command(
    url: Annotated[str, typer.Option(help="Public HTTPS PDF URL")],
    title: Annotated[str, typer.Option(help="Human-readable document title")],
    asset_id: Annotated[
        str | None,
        typer.Option(help="Optional stable lowercase asset ID"),
    ] = None,
    asset_type: Annotated[
        str,
        typer.Option("--type", help="Runtime asset type label"),
    ] = "manual_pdf",
    published_at: Annotated[
        str | None,
        typer.Option(
            "--published-at",
            help="Optional source publication timestamp with timezone",
        ),
    ] = None,
    metadata_path: Annotated[
        Path | None,
        typer.Option("--metadata-json", help="Optional administrator metadata JSON"),
    ] = None,
    result_path: Annotated[Path | None, typer.Option("--result")] = None,
) -> None:
    """Download one administrator-approved HTTPS PDF into immutable storage."""

    settings = Settings.from_env()
    settings.data_root.mkdir(parents=True, exist_ok=True)
    try:
        published = acquire_manual_https_pdf(
            settings.data_root,
            url=url,
            title=title,
            asset_id=asset_id,
            asset_type=asset_type,
            published_at=_parse_asset_timestamp(published_at),
            metadata=_read_admin_asset_metadata(metadata_path),
        )
    except (OSError, ValueError, ResearchAssetError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    result = {
        "status": "succeeded",
        "asset_id": published.asset_id,
        "kind": "pdf",
        "type": published.manifest.get("type"),
        "manifest_path": str(published.manifest_path),
    }
    _write_optional_result(result_path, result)
    console.print_json(json.dumps(result, ensure_ascii=False))


@app.command("research-assets")
def research_assets_command(
    snapshot_name: Annotated[
        str,
        typer.Option("--snapshot", help="Named verified immutable snapshot"),
    ],
    as_of: Annotated[
        str,
        typer.Option(help="Asia/Shanghai research day, YYYY-MM-DD or latest"),
    ] = "latest",
    tushare_report_date: Annotated[
        str | None,
        typer.Option(
            help="Tushare report day; defaults to latest PIT-eligible day in snapshot",
        ),
    ] = None,
    skip_tushare: Annotated[
        bool,
        typer.Option(help="Skip Tushare research_report PDF acquisition"),
    ] = False,
    skip_arxiv: Annotated[
        bool,
        typer.Option(help="Skip arXiv q-fin/cs.LG/stat.ML discovery"),
    ] = False,
    result_path: Annotated[Path | None, typer.Option("--result")] = None,
) -> None:
    """Safely acquire governed research PDFs from one immutable snapshot."""

    if skip_tushare and skip_arxiv:
        raise typer.BadParameter("at least one research asset source must be enabled")
    as_of_date = parse_date(as_of, latest=today_cn())
    report_date = parse_date(tushare_report_date) if tushare_report_date else None
    settings = Settings.from_env()
    settings.data_root.mkdir(parents=True, exist_ok=True)
    try:
        summary = ingest_research_assets(
            settings.data_root,
            snapshot_name=snapshot_name,
            as_of=as_of_date,
            include_tushare=not skip_tushare,
            include_arxiv=not skip_arxiv,
            tushare_report_date=report_date,
        )
    except (OSError, ValueError, ResearchAssetError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    result = summary.as_dict()
    result.update(
        {
            "as_of": as_of_date.isoformat(),
            "tushare_report_date": (
                report_date.isoformat() if report_date else "latest_eligible"
            ),
        }
    )
    _write_optional_result(result_path, result)
    console.print_json(json.dumps(result, ensure_ascii=False))
    if summary.failed:
        raise typer.Exit(3)


@app.command("report-rc-factors")
def report_rc_factors_command(
    ts_code: Annotated[str, typer.Option(help="Comma-separated Tushare codes to include")] = "",
    start: Annotated[str, typer.Option(help="YYYY-MM-DD report date")] = "2010-01-01",
    end: Annotated[str, typer.Option(help="YYYY-MM-DD or latest")] = "latest",
    result_path: Annotated[Path | None, typer.Option("--result")] = None,
) -> None:
    """Build structured factor artifacts from the downloaded report_rc dataset."""
    start_date = parse_date(start)
    end_date = parse_date(end, latest=today_cn())
    if end_date < start_date:
        raise typer.BadParameter("end must not be before start")
    context = load_context(
        require_credentials=False,
        progress_path=result_path,
        progress_target={
            "kind": "report_rc_factors",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        },
    )
    context.report_progress(
        "processing", "report_rc structured factor production", {"report_rc"}, force=True
    )
    summary = _produce_factors(
        "report-rc-factors",
        lambda: process_report_rc(
            context.settings.data_root,
            ts_codes=set(_split_codes(ts_code)) or None,
            start=start_date,
            end=end_date,
        ),
    )
    result = {
        "dataset": "report_rc",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "ts_codes": _split_codes(ts_code),
        **summary.as_dict(),
    }
    _write_optional_result(result_path, result)
    console.print_json(json.dumps(result, ensure_ascii=False))



@app.command("global-reference-factors")
def global_reference_factors_command(
    start: Annotated[str, typer.Option(help="YYYY-MM-DD source session date")] = "2008-01-01",
    end: Annotated[str, typer.Option(help="YYYY-MM-DD or latest")] = "latest",
    result_path: Annotated[Path | None, typer.Option("--result")] = None,
) -> None:
    """Build structured factor artifacts from peripheral global-reference datasets."""
    start_date = parse_date(start)
    end_date = parse_date(end, latest=today_cn())
    if end_date < start_date:
        raise typer.BadParameter("end must not be before start")
    context = load_context(
        require_credentials=False,
        progress_path=result_path,
        progress_target={
            "kind": "global_reference_factors",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        },
    )
    context.report_progress(
        "processing",
        "global-reference structured factor production",
        {"index_global", "us_daily", "us_tycr", "trade_cal"},
        force=True,
    )
    summary = _produce_factors(
        "global-reference-factors",
        lambda: process_global_reference(
            context.settings.data_root,
            start=start_date,
            end=end_date,
        ),
    )
    result = {
        "dataset": "global_reference",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        **summary.as_dict(),
    }
    _write_optional_result(result_path, result)
    console.print_json(json.dumps(result, ensure_ascii=False))


@app.command("major-news-mentions")
def major_news_mentions_command(
    ts_code: Annotated[str, typer.Option(help="Comma-separated Tushare codes to include")] = "",
    start: Annotated[str, typer.Option(help="YYYY-MM-DD publication date")] = "2024-01-01",
    end: Annotated[str, typer.Option(help="YYYY-MM-DD or latest")] = "latest",
    result_path: Annotated[Path | None, typer.Option("--result")] = None,
) -> None:
    """Map major_news mentions onto instruments and build mention factor artifacts."""
    start_date = parse_date(start)
    end_date = parse_date(end, latest=today_cn())
    if end_date < start_date:
        raise typer.BadParameter("end must not be before start")
    context = load_context(
        require_credentials=False,
        progress_path=result_path,
        progress_target={
            "kind": "major_news_mentions",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        },
    )
    context.report_progress("processing", "major_news mention mapping", {"major_news"}, force=True)
    summary = _produce_factors(
        "major-news-mentions",
        lambda: process_major_news_mentions(
            context.settings.data_root,
            ts_codes=set(_split_codes(ts_code)) or None,
            start=start_date,
            end=end_date,
        ),
    )
    result = {
        "dataset": "major_news",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "ts_codes": _split_codes(ts_code),
        **summary.as_dict(),
    }
    _write_optional_result(result_path, result)
    console.print_json(json.dumps(result, ensure_ascii=False))


@app.command("news-flash-factors")
def news_flash_factors_command(
    start: Annotated[str, typer.Option(help="YYYY-MM-DD publication date")] = "2018-11-20",
    end: Annotated[str, typer.Option(help="YYYY-MM-DD or latest")] = "latest",
    result_path: Annotated[Path | None, typer.Option("--result")] = None,
) -> None:
    """Build the market-level news-flash intensity factor artifact."""
    start_date = parse_date(start)
    end_date = parse_date(end, latest=today_cn())
    if end_date < start_date:
        raise typer.BadParameter("end must not be before start")
    context = load_context(
        require_credentials=False,
        progress_path=result_path,
        progress_target={
            "kind": "news_flash_factors",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        },
    )
    context.report_progress(
        "processing", "news flash intensity factor production", {"news"}, force=True
    )
    summary = _produce_factors(
        "news-flash-factors",
        lambda: process_news_flash(context.settings.data_root, start=start_date, end=end_date),
    )
    result = {
        "dataset": "news",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        **summary.as_dict(),
    }
    _write_optional_result(result_path, result)
    console.print_json(json.dumps(result, ensure_ascii=False))


def _historical_a_share_symbols(master: pd.DataFrame, *, start: date, end: date) -> list[str]:
    return sorted(_historical_a_share_active_ranges(master, start=start, end=end))


def _historical_a_share_active_ranges(
    master: pd.DataFrame, *, start: date, end: date
) -> dict[str, tuple[date, date]]:
    required = {"ts_code", "list_date", "delist_date"}
    if master.empty or not required <= set(master.columns):
        raise RuntimeError(
            "stock_basic lifecycle master is unavailable; run bootstrap reference first"
        )
    list_dates = pd.to_datetime(master["list_date"], format="%Y%m%d", errors="coerce")
    delist_dates = pd.to_datetime(master["delist_date"], format="%Y%m%d", errors="coerce")
    active = master.loc[
        list_dates.le(pd.Timestamp(end))
        & (delist_dates.isna() | delist_dates.ge(pd.Timestamp(start)))
        & master["ts_code"].fillna("").astype(str).str.upper().str.endswith((".SH", ".SZ", ".BJ"))
    ]
    ranges: dict[str, tuple[date, date]] = {}
    for index, row in active.iterrows():
        symbol = str(row["ts_code"]).strip().upper()
        listed_at = list_dates.loc[index].date()
        delisted_value = delist_dates.loc[index]
        delisted_at = end if pd.isna(delisted_value) else delisted_value.date()
        clipped = (max(start, listed_at), min(end, delisted_at))
        if clipped[1] < clipped[0]:
            continue
        previous = ranges.get(symbol)
        ranges[symbol] = (
            (
                min(previous[0], clipped[0]),
                max(previous[1], clipped[1]),
            )
            if previous
            else clipped
        )
    return ranges


def _historically_active_symbols(
    master: pd.DataFrame,
    *,
    start: date,
    end: date,
    suffixes: tuple[str, ...],
) -> list[str]:
    """Select master rows whose listing lifecycle intersects the request range."""

    if master.empty or "ts_code" not in master.columns:
        raise RuntimeError("market master is unavailable or missing ts_code")
    frame = master.copy()
    symbols = frame["ts_code"].fillna("").astype(str).str.strip().str.upper()
    mask = symbols.ne("")
    if suffixes:
        mask &= symbols.str.endswith(suffixes)
    if "list_date" in frame.columns:
        listed = pd.to_datetime(frame["list_date"], errors="coerce")
        mask &= listed.isna() | listed.le(pd.Timestamp(end))
    if "delist_date" in frame.columns:
        delisted = pd.to_datetime(frame["delist_date"], errors="coerce")
        mask &= delisted.isna() | delisted.ge(pd.Timestamp(start))
    return sorted(set(symbols.loc[mask].tolist()))


def _open_market_dates(calendar: pd.DataFrame, *, start: date, end: date) -> list[str]:
    if calendar.empty or "is_open" not in calendar.columns:
        raise RuntimeError("market trade calendar is unavailable or missing is_open")
    date_field = "cal_date" if "cal_date" in calendar.columns else "date"
    if date_field not in calendar.columns:
        raise RuntimeError("market trade calendar is missing cal_date/date")
    values = pd.to_datetime(calendar[date_field], errors="coerce")
    is_open = calendar["is_open"].astype(str).str.lower().isin({"1", "true", "t", "yes"})
    selected = is_open & values.between(pd.Timestamp(start), pd.Timestamp(end))
    return sorted(values.loc[selected].dt.strftime("%Y%m%d").dropna().unique().tolist())


def _index_active_ranges(
    master: pd.DataFrame, *, suffix: str, start: date, end: date
) -> dict[str, tuple[date, date]]:
    if master.empty or "ts_code" not in master.columns:
        raise RuntimeError("index_basic is unavailable; run the reference bootstrap first")
    symbols = master["ts_code"].fillna("").astype(str).str.strip().str.upper()
    selected = master.loc[symbols.str.endswith(suffix.upper())].copy()
    selected["_symbol"] = symbols.loc[selected.index]
    listed = pd.to_datetime(selected.get("list_date"), errors="coerce")
    expired = pd.to_datetime(selected.get("exp_date"), errors="coerce")
    ranges: dict[str, tuple[date, date]] = {}
    for index, row in selected.iterrows():
        listed_value = listed.loc[index]
        expired_value = expired.loc[index]
        symbol_start = start if pd.isna(listed_value) else max(start, listed_value.date())
        symbol_end = end if pd.isna(expired_value) else min(end, expired_value.date())
        if symbol_start <= symbol_end:
            ranges[str(row["_symbol"])] = (symbol_start, symbol_end)
    return dict(sorted(ranges.items()))


def _supersede_obsolete_derivative_units(context: Context) -> None:
    obsolete = context.checkpoint.unfinished_units({"fut_index_daily", "bc_bestotcqt"})
    if obsolete:
        context.checkpoint.supersede_units(
            [str(row["unit_key"]) for row in obsolete],
            (
                "obsolete derivative plan: NH indices require per-symbol index_daily requests; "
                "bc_bestotcqt returns anonymous prices without an auditable instrument key"
            ),
        )


@app.command("research-report-download")
def research_report_download(
    start: Annotated[str, typer.Option(help="YYYY-MM-DD")] = "2017-01-01",
    end: Annotated[str, typer.Option(help="YYYY-MM-DD or latest")] = "latest",
    result_path: Annotated[Path | None, typer.Option("--result")] = None,
) -> None:
    """Download only Tushare research_report for the isolated asset pipeline."""

    start_date = parse_date(start)
    end_date = parse_date(end, latest=today_cn())
    if end_date < start_date:
        raise typer.BadParameter("end must not be before start")
    context = load_context(
        progress_path=result_path,
        progress_target={
            "kind": "research_report_download",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        },
    )
    specs = [
        spec
        for spec in supplemental_specs(
            "research_corpus",
            start=start_date,
            end=end_date,
            trading_dates=(),
            max_attempts=context.settings.max_request_attempts,
        )
        if spec.dataset == "research_report"
    ]
    if not specs:
        raise RuntimeError("research_report planner produced no acquisition units")
    planned, rows, inserted = _run_paginated_specs(
        context,
        "isolated research_report acquisition",
        specs,
    )
    downloaded_rows = sum(int(row.get("row_count") or 0) for row in rows)
    if downloaded_rows == 0:
        raise RuntimeError(
            "research_report returned no metadata; provider entitlement and "
            "downloadable PDF coverage are not proven"
        )
    result = {
        "status": "succeeded",
        "dataset": "research_report",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "units": len(rows),
        "rows": downloaded_rows,
        "planned_units": len(planned),
        "newly_inserted": inserted,
        "next_step": "snapshot --profile research-assets",
    }
    _write_optional_result(result_path, result)
    console.print_json(json.dumps(result, ensure_ascii=False))


@app.command("supplemental-download")
def supplemental_download(
    bundle: Annotated[str, typer.Option(help="Independent supplemental data bundle")],
    start: Annotated[str, typer.Option(help="YYYY-MM-DD")] = "2024-01-01",
    end: Annotated[str, typer.Option(help="YYYY-MM-DD or latest")] = "latest",
    symbols: Annotated[
        str,
        typer.Option(help="Optional comma-separated HK/US financial universe"),
    ] = "",
    result_path: Annotated[Path | None, typer.Option("--result")] = None,
) -> None:
    """Download one independently resumable market or macro data bundle."""
    if bundle not in SUPPORTED_BUNDLES:
        raise typer.BadParameter("bundle must be one of: " + ", ".join(sorted(SUPPORTED_BUNDLES)))
    start_date = parse_date(start)
    end_date = parse_date(end, latest=today_cn())
    if end_date < start_date:
        raise typer.BadParameter("end must not be before start")
    context = load_context(
        progress_path=result_path,
        progress_target={
            "kind": "supplemental_download",
            "bundle": bundle,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "requested_symbols": len(_split_codes(symbols)),
        },
    )
    trading_dates = (
        context.planner.trading_dates(start_date, end_date)
        if bundle.startswith("cn_") and bundle != "cn_macro"
        else []
    )
    if bundle == "cn_institutional":
        specs = _institutional_history_specs(
            context,
            start=start_date,
            end=end_date,
            trading_dates=trading_dates,
            max_attempts=context.settings.max_request_attempts,
        )
    else:
        specs = supplemental_specs(
            bundle,
            start=start_date,
            end=end_date,
            trading_dates=trading_dates,
            max_attempts=context.settings.max_request_attempts,
        )
    datasets = {spec.dataset for spec in specs}
    context.report_progress("planning", f"{bundle} planning", datasets, force=True)
    rows: list[dict] = []
    inserted = 0
    if bundle == "cn_governance_risk":
        _supersede_unsupported_governance_units(context)
    if bundle == "cn_derivatives_enhanced":
        _supersede_obsolete_derivative_units(context)
    if bundle == "research_corpus":
        _supersede_unsupported_research_units(context)
    if specs:
        specs, rows, inserted = _run_paginated_specs(context, bundle, specs)
    console.print(
        f"planned supplemental bundle={bundle} units={len(specs)} newly_inserted={inserted}"
    )
    if bundle == "cn_governance_risk":
        master = context.storage.read_units(context.checkpoint.successful("stock_basic"))
        active_ranges = _historical_a_share_active_ranges(
            master,
            start=start_date,
            end=end_date,
        )
        cyq_ranges = {
            symbol: active_range
            for symbol, active_range in active_ranges.items()
            if active_range[1] >= date(2018, 1, 1)
        }
        secondary_specs = coverage_secondary_specs(
            bundle,
            {
                "stk_rewards": active_ranges,
                "cyq_perf": cyq_ranges,
            },
            start=start_date,
            end=end_date,
            max_attempts=context.settings.max_request_attempts,
        )
        secondary_specs, secondary_rows, secondary_inserted = _run_paginated_specs(
            context,
            f"{bundle} symbol data",
            secondary_specs,
        )
        inserted += secondary_inserted
        secondary_datasets = {spec.dataset for spec in secondary_specs}
        specs.extend(secondary_specs)
        rows.extend(secondary_rows)
        datasets.update(secondary_datasets)
    if bundle == "cn_derivatives_enhanced":
        current_index_units = select_current_reference_units(
            context.checkpoint.successful("index_basic"),
            snapshot_end=end_date,
        )
        index_master = context.storage.read_units(current_index_units)
        index_ranges = _index_active_ranges(
            index_master,
            suffix=".NH",
            start=start_date,
            end=end_date,
        )
        if not index_ranges:
            raise RuntimeError(
                "index_basic has no .NH contracts; rerun the core reference bootstrap"
            )
        secondary_specs = coverage_secondary_specs(
            bundle,
            {"fut_index_daily": index_ranges},
            start=start_date,
            end=end_date,
            max_attempts=context.settings.max_request_attempts,
        )
        secondary_specs, secondary_rows, secondary_inserted = _run_paginated_specs(
            context,
            f"{bundle} NH index history",
            secondary_specs,
        )
        inserted += secondary_inserted
        secondary_datasets = {spec.dataset for spec in secondary_specs}
        specs.extend(secondary_specs)
        rows.extend(secondary_rows)
        datasets.update(secondary_datasets)
    if bundle == "strategy_specialty_minutes":
        requested = _split_codes(symbols)
        if not requested:
            raise typer.BadParameter(
                "strategy_specialty_minutes requires --symbols; HK codes are routed "
                "to hk_mins and all other codes to sw_mins"
            )
        secondary_specs = coverage_secondary_specs(
            bundle,
            {
                "hk_mins": [value for value in requested if value.upper().endswith(".HK")],
                "sw_mins": [value for value in requested if not value.upper().endswith(".HK")],
            },
            start=start_date,
            end=end_date,
            max_attempts=context.settings.max_request_attempts,
        )
        inserted += context.checkpoint.add(secondary_specs)
        context.checkpoint.retry_failed_units(spec.unit_key for spec in secondary_specs)
        secondary_datasets = {spec.dataset for spec in secondary_specs}
        _run_phase(context, f"{bundle} symbol windows", secondary_datasets)
        secondary_rows = _require_specs_complete(context, secondary_specs)
        specs.extend(secondary_specs)
        rows.extend(secondary_rows)
        datasets.update(secondary_datasets)
    if bundle == "cn_options_bonds":
        master = context.storage.read_units(context.checkpoint.successful("cb_basic"))
        if "ts_code" not in master.columns:
            raise RuntimeError("cb_basic did not provide ts_code for bond reference planning")
        bond_symbols = sorted(
            {
                str(value).strip()
                for value in master["ts_code"].dropna().tolist()
                if str(value).strip()
            }
        )
        if not bond_symbols:
            raise RuntimeError("cb_basic produced an empty convertible-bond universe")
        reference_specs = bond_reference_specs(
            bond_symbols,
            start=start_date,
            end=end_date,
            max_attempts=context.settings.max_request_attempts,
        )
        inserted += context.checkpoint.add(reference_specs)
        context.checkpoint.retry_failed_units(spec.unit_key for spec in reference_specs)
        reference_datasets = {spec.dataset for spec in reference_specs}
        _run_phase(context, f"{bundle} references", reference_datasets)
        reference_rows = _require_specs_complete(context, reference_specs)
        specs.extend(reference_specs)
        rows.extend(reference_rows)
    if bundle in {"hk_market", "us_market"}:
        market = bundle.split("_", 1)[0]
        basic_dataset = f"{market}_basic"
        calendar_dataset = f"{market}_tradecal"
        calendar = context.storage.read_units(context.checkpoint.successful(calendar_dataset))
        open_dates = _open_market_dates(calendar, start=start_date, end=end_date)
        daily_specs = market_daily_specs(
            market,
            open_dates,
            max_attempts=context.settings.max_request_attempts,
        )
        daily_specs, daily_rows, daily_inserted = _run_paginated_specs(
            context, f"{bundle} open-session prices", daily_specs
        )
        inserted += daily_inserted
        specs.extend(daily_specs)
        rows.extend(daily_rows)
        datasets.update(spec.dataset for spec in daily_specs)
        financial_symbols = _split_codes(symbols)
        if not financial_symbols:
            master = context.storage.read_units(context.checkpoint.successful(basic_dataset))
            if "ts_code" not in master.columns:
                raise RuntimeError(
                    f"{basic_dataset} did not provide ts_code for financial planning"
                )
            financial_symbols = _historically_active_symbols(
                master,
                start=start_date,
                end=end_date,
                suffixes=((".HK",) if market == "hk" else ()),
            )
        if not financial_symbols:
            raise RuntimeError(f"{basic_dataset} produced an empty financial universe")
        financial_specs = market_financial_specs(
            market,
            financial_symbols,
            start=start_date,
            end=end_date,
            max_attempts=context.settings.max_request_attempts,
        )
        financial_specs, financial_rows, financial_inserted = _run_paginated_specs(
            context, f"{bundle} financials", financial_specs
        )
        inserted += financial_inserted
        financial_datasets = {spec.dataset for spec in financial_specs}
        specs.extend(financial_specs)
        rows.extend(financial_rows)
        datasets.update(financial_datasets)
    by_dataset: dict[str, dict[str, int]] = {}
    keys_by_dataset = {
        dataset: {spec.unit_key for spec in specs if spec.dataset == dataset}
        for dataset in sorted(datasets)
    }
    for dataset, keys in keys_by_dataset.items():
        selected = [row for row in rows if str(row["unit_key"]) in keys]
        by_dataset[dataset] = {
            "units": len(selected),
            "rows": sum(int(row.get("row_count") or 0) for row in selected),
        }
    result = {
        "status": "succeeded",
        "bundle": bundle,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "units": len(rows),
        "rows": sum(int(row.get("row_count") or 0) for row in rows),
        "datasets": by_dataset,
        "pagination_verified": True,
        "requested_symbols": _split_codes(symbols),
    }
    _write_optional_result(result_path, result)
    console.print_json(json.dumps(result, ensure_ascii=False))


@app.command("build-qlib")
def build_qlib_command(
    snapshot_name: Annotated[str | None, typer.Option("--snapshot")] = None,
    staging_only: Annotated[bool, typer.Option("--staging-only")] = False,
    skip_quality_gate: Annotated[
        bool,
        typer.Option(
            "--skip-quality-gate",
            help="Allow an unverified staging-only diagnostic build; never publishes Qlib data",
        ),
    ] = False,
) -> None:
    """Normalize a Parquet snapshot and build Qlib binary data."""
    context = load_context(require_credentials=False)
    if snapshot_name:
        snapshot_path = context.storage.snapshots_root / snapshot_name
    else:
        candidates = sorted(
            path for path in context.storage.snapshots_root.glob("*") if path.is_dir()
        )
        if not candidates:
            raise typer.BadParameter("no snapshots are available")
        snapshot_path = candidates[-1]
    result = _build_qlib(
        context, snapshot_path, staging_only=staging_only, skip_quality_gate=skip_quality_gate
    )
    console.print(result)


@app.command("build-minute-qlib")
def build_minute_qlib_command(
    snapshot_name: Annotated[str, typer.Option("--snapshot")],
    output_name: Annotated[str | None, typer.Option("--output-name")] = None,
    expected_manifest_sha256: Annotated[
        str | None,
        typer.Option(
            "--expected-manifest-sha256",
            help="Manifest digest sealed when the build job was created",
        ),
    ] = None,
    target_frequency: Annotated[str | None, typer.Option("--target-frequency")] = None,
    staging_only: Annotated[bool, typer.Option("--staging-only")] = False,
    skip_quality_gate: Annotated[
        bool,
        typer.Option(
            "--skip-quality-gate",
            help="Allow an unverified staging-only diagnostic build; never publishes Qlib data",
        ),
    ] = False,
) -> None:
    """Build native or Qlib-resampled data without another download path."""
    context = load_context(require_credentials=False)
    snapshot_path = context.storage.snapshots_root / snapshot_name
    if expected_manifest_sha256 is not None:
        if not _is_sha256(expected_manifest_sha256):
            raise typer.BadParameter("expected snapshot manifest SHA-256 is invalid")
        try:
            actual_manifest_sha256 = hashlib.sha256(
                (snapshot_path / "manifest.json").read_bytes()
            ).hexdigest()
        except OSError as exc:
            raise typer.BadParameter("snapshot manifest is missing") from exc
        if actual_manifest_sha256 != expected_manifest_sha256.lower():
            raise typer.BadParameter(
                "snapshot manifest changed after the minute Qlib job was sealed"
            )
    result = _build_minute_qlib(
        context,
        snapshot_path,
        output_name=output_name,
        target_frequency=target_frequency,
        staging_only=staging_only,
        skip_quality_gate=skip_quality_gate,
    )
    console.print(result)


def _require_snapshot_quality_gate(snapshot_path: Path, *, skip: bool = False) -> None:
    """Refuse Qlib builds for snapshots whose manifest has no passing quality gate."""

    if skip:
        return
    try:
        manifest = json.loads((snapshot_path / "manifest.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"snapshot manifest is missing or invalid: {snapshot_path}") from exc
    gate = manifest.get("quality_gate")
    if not isinstance(gate, dict) or gate.get("ok") is not True:
        raise ValueError(
            "snapshot has no passing quality gate; rebuild it through the verify "
            "and snapshot commands"
        )


def _build_snapshot(
    context: Context,
    name: str,
    start_date: date,
    end_date: date,
    profile: str,
    quality_gate: dict[str, Any] | None = None,
    industry_history_anchor: str | None = None,
) -> Path:
    if not quality_gate or quality_gate.get("ok") is not True:
        raise ValueError("snapshot publication requires a passing bound quality gate")
    module_root = Path(__file__).resolve().parent
    is_research_asset_source = profile == RESEARCH_ASSET_SNAPSHOT_PROFILE
    is_global_reference_source = profile == GLOBAL_REFERENCE_SNAPSHOT_PROFILE
    mixed_legacy_source = (
        not is_research_asset_source
        and not is_global_reference_source
        and start_date < date(2016, 1, 1)
    )
    provider_contract = (
        "tushare-compatible+baostock-audited-legacy"
        if mixed_legacy_source
        else "tushare-compatible"
    )
    contract_files = {
        "planner": module_root / "planner.py",
        "provider": module_root / "provider.py",
        "storage": module_root / "storage.py",
    }
    if is_research_asset_source:
        contract_files.update(
            {
                "catalog": module_root / "catalog.py",
                "research_assets": module_root / "research_assets.py",
            }
        )
    if is_global_reference_source:
        contract_files.update(
            {
                "catalog": module_root / "catalog.py",
                "availability": module_root / "availability.py",
            }
        )
    if mixed_legacy_source:
        contract_files.update(
            {
                "baostock_provider": module_root / "baostock_provider.py",
                "legacy_market": module_root / "legacy_market.py",
            }
        )
    lineage_configuration = {
        "profile": profile,
        "start_date": start_date.isoformat(),
        "provider": provider_contract,
        "legacy_source": BAOSTOCK_SOURCE_VERSION if mixed_legacy_source else None,
        "legacy_overlap_policy_version": (
            BAOSTOCK_OVERLAP_POLICY_VERSION if mixed_legacy_source else None
        ),
        "industry_history_carry_rule_version": (
            INDUSTRY_HISTORY_CARRY_RULE_VERSION
            if not is_research_asset_source and not is_global_reference_source
            else None
        ),
        "ingestion_contract_sha256": file_contract_sha256(contract_files),
    }
    lineage_contract = {
        "kind": (
            "research_asset_source"
            if is_research_asset_source
            else (
                "global_reference_source"
                if is_global_reference_source
                else "qlib_daily_source"
            )
        ),
        "configuration": lineage_configuration,
    }
    lineage_id = make_lineage_id(
        lineage_contract["kind"],
        lineage_configuration,
    )
    industry_anchor_path: Path | None = None
    industry_anchor_evidence: dict[str, Any] | None = None
    if industry_history_anchor is not None:
        if is_research_asset_source or is_global_reference_source:
            raise ValueError(
                "industry history anchor is only valid for A-share snapshots"
            )
        if industry_history_anchor == name:
            raise ValueError("industry history anchor must differ from the target snapshot")
        industry_anchor_path, industry_anchor_evidence = (
            resolve_verified_snapshot_anchor(
                context.storage.snapshots_root,
                industry_history_anchor,
                required_profile=profile,
                required_start=start_date,
                maximum_end=end_date,
                required_dataset="index_member_all",
            )
        )
    existing = context.storage.snapshots_root / name
    if existing.exists():
        manifest_path = existing / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise ValueError(f"existing snapshot {name!r} is incomplete") from exc
        expected = {
            "profile": profile,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "lineage_id": lineage_id,
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError(f"existing snapshot {name!r} does not match the requested range")
        if manifest.get("industry_history_anchor") != industry_anchor_evidence:
            raise ValueError(
                f"existing snapshot {name!r} does not match the requested industry anchor"
            )
        if quality_gate:
            gate_scope = quality_gate.get("plan_scope_sha256") or quality_gate.get(
                "release_window_scope_sha256"
            )
            if manifest.get("plan_scope_sha256") != gate_scope:
                raise ValueError(
                    f"existing snapshot {name!r} does not match the verified plan scope"
                )
            if manifest.get("selected_unit_set_sha256") != quality_gate.get(
                "selected_unit_set_sha256"
            ):
                raise ValueError(
                    f"existing snapshot {name!r} does not match the verified unit set"
                )
        return existing
    available = set(context.checkpoint.datasets())
    selected_datasets = _snapshot_datasets(profile, available=available)
    selection = select_release_window_units(
        context.checkpoint.active_units(selected_datasets & available),
        snapshot_start=start_date,
        snapshot_end=end_date,
        datasets=selected_datasets,
        profile=profile,
    )
    incomplete = [
        row for row in selection.rows if str(row.get("status") or "") != "succeeded"
    ]
    if incomplete:
        raise ValueError(
            f"{len(incomplete)} selected release-window work units are incomplete"
        )
    if quality_gate:
        verified_scope = quality_gate.get("plan_scope_sha256") or quality_gate.get(
            "release_window_scope_sha256"
        )
        if verified_scope != selection.plan_scope_sha256:
            raise ValueError(
                "snapshot release window no longer matches its verification scope"
            )
        if (
            quality_gate.get("selected_unit_set_sha256")
            != selection.selected_unit_set_sha256
        ):
            raise ValueError(
                "snapshot selected work units changed after quality verification"
            )
    units: dict[str, list[dict[str, Any]]] = {}
    for row in selection.rows:
        units.setdefault(str(row["dataset"]), []).append(dict(row))
    if not units:
        raise ValueError(f"no successful {profile} datasets are available for snapshotting")
    lineage = prepare_lineage_metadata(
        context.storage.snapshots_root,
        lineage_id=lineage_id,
        end_date=end_date,
        successful_units=units,
    )
    if industry_anchor_path is not None and lineage.get("parent_snapshot") is not None:
        raise ValueError(
            "industry history anchor is only allowed for a new lineage root; "
            "a compatible parent snapshot already exists"
        )
    base_snapshot: Path | None = None
    parent_name = lineage.get("parent_snapshot")
    if parent_name:
        parent_path = context.storage.snapshots_root / str(parent_name)
        if (parent_path / "manifest.json").exists():
            base_snapshot = parent_path
    return context.storage.build_snapshot(
        name=name,
        successful_units=units,
        manifest_extra={
            "profile": profile,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "plan_scope_sha256": selection.plan_scope_sha256,
            "selected_unit_set_sha256": selection.selected_unit_set_sha256,
            "selected_unit_evidence": [
                dict(item) for item in selection.unit_identities
            ],
            "provider": provider_contract,
            "lineage_contract": lineage_contract,
            "source_contracts": (
                {
                    "primary": "tushare-compatible",
                    "legacy_market": BAOSTOCK_SOURCE_VERSION,
                    "legacy_overlap_policy_version": BAOSTOCK_OVERLAP_POLICY_VERSION,
                }
                if mixed_legacy_source
                else {"primary": "tushare-compatible"}
            ),
            **({"quality_gate": quality_gate} if quality_gate else {}),
            **(
                {"industry_history_anchor": industry_anchor_evidence}
                if industry_anchor_evidence is not None
                else {}
            ),
            **lineage,
        },
        base_snapshot=base_snapshot,
        industry_history_anchor=industry_anchor_path,
    )


def _build_qlib(
    context: Context,
    snapshot_path: Path,
    *,
    staging_only: bool,
    skip_quality_gate: bool = False,
) -> Path:
    if skip_quality_gate and not staging_only:
        raise ValueError(
            "an unverified snapshot may only be normalized with --staging-only; "
            "publishing Qlib binaries requires a passing quality gate"
        )
    _require_snapshot_quality_gate(snapshot_path, skip=skip_quality_gate)
    try:
        snapshot_manifest = json.loads(
            (snapshot_path / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("snapshot manifest is missing or invalid") from exc
    if snapshot_manifest.get("profile") == RESEARCH_ASSET_SNAPSHOT_PROFILE:
        raise ValueError(
            "research-assets snapshots are isolated PDF acquisition sources and "
            "cannot be normalized into a Qlib market dataset"
        )
    if snapshot_manifest.get("profile") == GLOBAL_REFERENCE_SNAPSHOT_PROFILE:
        raise ValueError(
            "global-reference snapshots are isolated peripheral-market sources and "
            "cannot be normalized into a Qlib market dataset"
        )
    # Production Qlib artifacts must carry the governed domestic-equity ETF
    # whitelist.  Direct QlibBuilder construction remains usable for isolated
    # forensic/unit fixtures that intentionally contain only A-share inputs.
    builder = QlibBuilder(snapshot_path, require_governed_etfs=True)
    staging = context.settings.data_root / "qlib_staging" / snapshot_path.name
    output = context.settings.data_root / "qlib" / snapshot_path.name
    if not staging_only and output.exists():
        required = (
            output / "calendars" / "day.txt",
            output / "instruments" / "cn_all.txt",
            output / "features",
            output / "metadata" / "provenance.json",
        )
        if all(path.exists() for path in required) and any(
            (output / "features").rglob("*.day.bin")
        ):
            provenance_path = output / "metadata" / "provenance.json"
            try:
                provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
                snapshot_manifest = verify_snapshot_lineage(snapshot_path)
                snapshot_manifest_sha256 = builder._snapshot_manifest_digest()
                require_daily_qlib_contract(provenance)
                verify_qlib_output_manifest(output, provenance)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"existing daily Qlib provenance is invalid: {output}"
                ) from exc
            expected = {
                "snapshot_name": snapshot_path.name,
                "snapshot_manifest_sha256": snapshot_manifest_sha256,
                "qlib_builder_sha256": builder.builder_sha256(),
                "source_lineage_id": snapshot_manifest.get("lineage_id"),
            }
            if (
                all(provenance.get(key) == value for key, value in expected.items())
                and _is_sha256(provenance.get("dataset_identity_sha256"))
                and _is_sha256(provenance.get("dataset_lineage_id"))
            ):
                return output
            raise ValueError(
                f"existing daily Qlib output belongs to different inputs or an "
                f"obsolete builder contract: {output}"
            )
        raise ValueError(
            f"existing Qlib output is incomplete and requires operator review: {output}"
        )
    by_symbol = builder.build_staging(staging)
    if staging_only:
        return by_symbol
    return builder.dump_bin(
        staging_by_symbol=by_symbol,
        qlib_dir=output,
        qlib_repo=context.settings.qlib_repo,
        qlib_python=context.settings.qlib_python,
        wsl_distro=context.settings.qlib_wsl_distro,
        max_workers=DAILY_QLIB_DUMP_WORKERS,
    )


def _build_minute_qlib(
    context: Context,
    snapshot_path: Path,
    *,
    output_name: str | None,
    target_frequency: str | None = None,
    staging_only: bool,
    skip_quality_gate: bool = False,
) -> Path:
    if skip_quality_gate and not staging_only:
        raise ValueError(
            "an unverified snapshot may only be normalized with --staging-only; "
            "publishing minute Qlib binaries requires a passing quality gate"
        )
    _require_snapshot_quality_gate(snapshot_path, skip=skip_quality_gate)
    builder = MinuteQlibBuilder(snapshot_path, target_frequency=target_frequency)
    output_name = output_name or f"{snapshot_path.name}-{builder.frequency}"
    staging = context.settings.data_root / "qlib_staging" / output_name
    output = context.settings.data_root / "qlib" / output_name
    if not staging_only and output.exists():
        required = (
            output / "calendars" / f"{builder.frequency}.txt",
            output / "instruments" / "all.txt",
            output / "features",
            output / "metadata" / "provenance.json",
        )
        if all(path.exists() for path in required) and any(
            (output / "features").rglob(f"*.{builder.frequency}.bin")
        ):
            provenance_path = output / "metadata" / "provenance.json"
            try:
                provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
                verify_qlib_output_manifest(output, provenance)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"existing minute Qlib provenance is invalid: {output}"
                ) from exc
            snapshot_manifest_sha256 = hashlib.sha256(
                (snapshot_path / "manifest.json").read_bytes()
            ).hexdigest()
            expected = {
                "snapshot_name": snapshot_path.name,
                "snapshot_manifest_sha256": snapshot_manifest_sha256,
                "frequency": builder.frequency,
                "source_frequency": builder.source_frequency,
                "execution_contract_version": MINUTE_EXECUTION_CONTRACT_VERSION,
                "qlib_builder_sha256": builder.builder_sha256(),
                "source_lineage_id": builder.source_lineage_id,
                "source_lineage_evidence_sha256": builder.source_lineage_evidence[
                    "evidence_sha256"
                ],
            }
            if (
                all(provenance.get(key) == value for key, value in expected.items())
                and provenance.get("lineage_verified") is True
                and _is_sha256(provenance.get("dataset_identity_sha256"))
                and _is_sha256(provenance.get("dataset_lineage_id"))
            ):
                return output
            raise ValueError(
                f"existing minute Qlib output belongs to different inputs or an "
                f"obsolete contract: {output}"
            )
        raise ValueError(
            f"existing minute Qlib output is incomplete and requires operator review: {output}"
        )
    native_staging = (
        staging.with_name(f"{staging.name}.native") if builder.requires_resampling else staging
    )
    by_symbol = builder.build_staging(native_staging)
    if builder.requires_resampling:
        by_symbol = builder.resample_staging(
            native_by_symbol=by_symbol,
            staging_path=staging,
            qlib_python=context.settings.qlib_python,
            wsl_distro=context.settings.qlib_wsl_distro,
        )
    if staging_only:
        return by_symbol
    return builder.dump_bin(
        staging_by_symbol=by_symbol,
        qlib_dir=output,
        qlib_repo=context.settings.qlib_repo,
        qlib_python=context.settings.qlib_python,
        wsl_distro=context.settings.qlib_wsl_distro,
        max_workers=min(16, max(1, context.settings.workers * 2)),
    )


def _require_specs_complete(
    context: Context,
    specs: list[FetchSpec],
    *,
    hint: str | None = None,
) -> list[dict]:
    keys = {spec.unit_key for spec in specs}
    rows = context.checkpoint.successful_units(keys)
    missing = keys - {str(row["unit_key"]) for row in rows}
    if missing:
        suffix = f"; {hint}" if hint else ""
        raise RuntimeError(f"{len(missing)} required work units are incomplete{suffix}")
    return rows


def _is_sha256(value: object) -> bool:
    normalized = str(value or "").lower()
    return len(normalized) == 64 and all(
        character in "0123456789abcdef" for character in normalized
    )


def _require_local_daily_source_lineage(
    context: Context,
    *,
    source_lineage_id: str,
    daily_source_dataset: str | None,
    start_date: date,
    end_date: date,
) -> dict[str, Any]:
    """Bind an execution snapshot to a real, verified local daily Qlib build."""

    normalized_lineage = str(source_lineage_id).lower()
    if not _is_sha256(normalized_lineage):
        raise typer.BadParameter(
            "minute download requires a valid daily --source-lineage-id"
        )
    qlib_root = context.settings.data_root / "qlib"
    if daily_source_dataset and Path(daily_source_dataset).name != daily_source_dataset:
        raise typer.BadParameter("--daily-source-dataset must be a local dataset name")
    candidates: list[dict[str, Any]] = []
    failures: list[str] = []
    for qlib_path in sorted(
        (
            path
            for path in qlib_root.iterdir()
            if path.is_dir()
            and (not daily_source_dataset or path.name == daily_source_dataset)
        ),
        reverse=True,
    ) if qlib_root.exists() else []:
        provenance_path = qlib_path / "metadata" / "provenance.json"
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(provenance.get("source_lineage_id") or "").lower() != normalized_lineage:
            continue
        try:
            require_daily_qlib_contract(provenance)
            verify_qlib_output_manifest(qlib_path, provenance)
            for field in (
                "dataset_identity_sha256",
                "dataset_lineage_id",
                "snapshot_manifest_sha256",
            ):
                if not _is_sha256(provenance.get(field)):
                    raise ValueError(f"daily Qlib provenance has invalid {field}")
            features = qlib_path / "features"
            instruments = (
                qlib_path / "instruments" / "cn_all.txt",
                qlib_path / "instruments" / "liquid_all.txt",
                qlib_path / "instruments" / "all.txt",
            )
            if not features.is_dir() or not any(path.is_file() for path in instruments):
                raise ValueError("daily Qlib features or instruments are incomplete")
            calendar_path = qlib_path / "calendars" / "day.txt"
            calendar = [
                date.fromisoformat(line.strip())
                for line in calendar_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not calendar or calendar[0] > start_date or calendar[-1] < end_date:
                raise ValueError("daily Qlib calendar does not cover the minute request")
            snapshot_name = str(provenance.get("snapshot_name") or "")
            if not snapshot_name or Path(snapshot_name).name != snapshot_name:
                raise ValueError("daily Qlib source snapshot name is invalid")
            snapshot_path = context.settings.data_root / "snapshots" / snapshot_name
            manifest = verify_snapshot_lineage(snapshot_path)
            manifest_path = snapshot_path / "manifest.json"
            if str(manifest.get("lineage_id") or "").lower() != normalized_lineage:
                raise ValueError("daily Qlib and source snapshot lineage disagree")
            manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            if manifest_sha256 != provenance["snapshot_manifest_sha256"]:
                raise ValueError("daily Qlib source snapshot digest is stale")
        except (OSError, ValueError) as exc:
            failures.append(f"{qlib_path.name}: {exc}")
            continue
        evidence = {
            "qlib_dataset": qlib_path.name,
            "qlib_dataset_identity_sha256": provenance["dataset_identity_sha256"],
            "qlib_dataset_lineage_id": provenance["dataset_lineage_id"],
            "source_snapshot": snapshot_name,
            "source_snapshot_manifest_sha256": provenance["snapshot_manifest_sha256"],
            "source_lineage_id": normalized_lineage,
            "calendar_start": calendar[0].isoformat(),
            "calendar_end": calendar[-1].isoformat(),
        }
        evidence["evidence_sha256"] = hashlib.sha256(
            json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        candidates.append(evidence)
    if not candidates:
        detail = f" ({failures[0]})" if failures else ""
        raise typer.BadParameter(
            "--source-lineage-id does not match a verified local daily Qlib dataset "
            f"covering {start_date.isoformat()}..{end_date.isoformat()}{detail}"
        )
    return max(
        candidates,
        key=lambda item: (str(item["calendar_end"]), str(item["qlib_dataset"])),
    )


def _source_snapshot_dataset_rows(
    context: Context,
    *,
    source_lineage_evidence: dict[str, Any],
    dataset: str,
    start_date: date,
    end_date: date,
) -> list[dict[str, Any]]:
    """Resolve the exact checkpoint units sealed into a verified source snapshot."""

    snapshot_name = str(source_lineage_evidence.get("source_snapshot") or "")
    if not snapshot_name or Path(snapshot_name).name != snapshot_name:
        raise ValueError("execution source snapshot name is invalid")
    manifest = verify_snapshot_lineage(
        context.settings.data_root / "snapshots" / snapshot_name
    )
    entry = (manifest.get("datasets") or {}).get(dataset)
    identities = entry.get("source_units") if isinstance(entry, dict) else None
    if not isinstance(identities, list) or not identities:
        raise ValueError(f"daily source snapshot has no {dataset} units")
    expected = {
        (
            str(item.get("unit_key") or ""),
            str(item.get("sha256") or ""),
            int(item.get("row_count") or 0),
        )
        for item in identities
        if isinstance(item, dict)
    }
    if len(expected) != len(identities) or any(
        not unit_key or not _is_sha256(sha256)
        for unit_key, sha256, _ in expected
    ):
        raise ValueError(f"daily source snapshot {dataset} unit evidence is invalid")
    rows = context.checkpoint.successful_units(unit_key for unit_key, _, _ in expected)
    actual = {
        (
            str(row.get("unit_key") or ""),
            str(row.get("sha256") or ""),
            int(row.get("row_count") or 0),
        )
        for row in rows
        if str(row.get("dataset") or "") == dataset
    }
    if actual != expected:
        raise ValueError(
            f"checkpoint no longer matches the {dataset} units sealed by the daily source"
        )
    selection = select_release_window_units(
        rows,
        snapshot_start=start_date,
        snapshot_end=end_date,
        datasets={dataset},
        profile="execution_source",
    )
    selected_rows = [dict(row) for row in selection.rows]
    if not selected_rows:
        raise ValueError(
            f"daily source snapshot has no {dataset} units in the execution window"
        )
    data_root = context.settings.data_root.resolve()
    for row in selected_rows:
        output_path = str(row.get("output_path") or "")
        target = (data_root / output_path).resolve()
        try:
            target.relative_to(data_root)
        except ValueError as exc:
            raise ValueError(f"daily source {dataset} unit path is unsafe") from exc
        if not output_path or not target.is_file():
            raise ValueError(f"daily source {dataset} unit file is missing")
        if hashlib.sha256(target.read_bytes()).hexdigest() != str(
            row.get("sha256") or ""
        ).lower():
            raise ValueError(f"daily source {dataset} unit checksum failed")
    return selected_rows


def _source_daily_trading_dates(
    context: Context,
    *,
    source_lineage_evidence: dict[str, Any],
    start_date: date,
    end_date: date,
) -> list[str]:
    """Use the exact daily Qlib calendar and prove it matches its source snapshot."""

    calendar_rows = _source_snapshot_dataset_rows(
        context,
        source_lineage_evidence=source_lineage_evidence,
        dataset="trade_cal",
        start_date=start_date,
        end_date=end_date,
    )
    source_dates = _open_market_dates(
        context.storage.read_units(calendar_rows),
        start=start_date,
        end=end_date,
    )
    qlib_dataset = str(source_lineage_evidence.get("qlib_dataset") or "")
    if not qlib_dataset or Path(qlib_dataset).name != qlib_dataset:
        raise ValueError("daily source Qlib dataset name is invalid")
    calendar_path = (
        context.settings.data_root / "qlib" / qlib_dataset / "calendars" / "day.txt"
    )
    try:
        qlib_dates = sorted(
            {
                value.strftime("%Y%m%d")
                for value in (
                    date.fromisoformat(line.strip())
                    for line in calendar_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
                if start_date <= value <= end_date
            }
        )
    except (OSError, ValueError) as exc:
        raise ValueError("bound daily Qlib calendar is missing or invalid") from exc
    if not qlib_dates:
        raise ValueError("bound daily Qlib calendar has no sessions in the minute window")
    if qlib_dates != source_dates:
        raise ValueError("bound daily Qlib calendar disagrees with its source snapshot trade_cal")
    return qlib_dates


def _explicit_execution_quality_gate(
    context: Context,
    *,
    selected: dict[str, list[dict[str, Any]]],
    start_date: date,
    end_date: date,
    profile: str,
) -> dict[str, Any]:
    """Checksum and seal the exact bounded units used by an execution snapshot."""

    rows = [dict(row) for dataset_rows in selected.values() for row in dataset_rows]
    if not rows:
        raise ValueError("execution snapshot has no selected work units")
    unit_keys = [str(row.get("unit_key") or "") for row in rows]
    if any(not key for key in unit_keys) or len(unit_keys) != len(set(unit_keys)):
        raise ValueError("execution snapshot work-unit identities are missing or duplicated")
    data_root = context.settings.data_root.resolve()
    for row in rows:
        output_path = str(row.get("output_path") or "")
        target = (data_root / output_path).resolve()
        try:
            target.relative_to(data_root)
        except ValueError as exc:
            raise ValueError("execution snapshot contains an unsafe unit path") from exc
        if not output_path or not target.is_file():
            raise ValueError(f"execution work-unit file is missing: {row['unit_key']}")
        expected_sha256 = str(row.get("sha256") or "").lower()
        if not _is_sha256(expected_sha256) or (
            hashlib.sha256(target.read_bytes()).hexdigest() != expected_sha256
        ):
            raise ValueError(f"execution work-unit checksum failed: {row['unit_key']}")
    selection = select_release_window_units(
        rows,
        snapshot_start=start_date,
        snapshot_end=end_date,
        datasets=set(selected),
        profile=profile,
    )
    selected_keys = {str(row["unit_key"]) for row in selection.rows}
    if selected_keys != set(unit_keys):
        raise ValueError("execution quality gate and snapshot unit selections disagree")
    minute_source_audits: dict[str, dict[str, Any]] = {}
    minute_source_warnings: list[str] = []
    ashare_rows = selected.get("ashare_5m") or []
    if ashare_rows:
        ashare_paths = [
            (data_root / str(row["output_path"])).resolve()
            for row in ashare_rows
            if str(row.get("output_path") or "").endswith(".parquet")
        ]
        if not ashare_paths:
            raise ValueError("A-share five-minute selection has no non-empty Parquet units")
        source_errors, source_warnings, source_audit = verify_ashare_5m_source_files(
            ashare_paths,
            snapshot_start=start_date,
            snapshot_end=end_date,
        )
        if source_errors:
            raise ValueError("; ".join(source_errors))
        minute_source_audits["ashare_5m"] = source_audit
        minute_source_warnings.extend(source_warnings)
    result = quality_gate_payload(
        {
            "ok": True,
            "checked_at": datetime.now(UTC).isoformat(),
            "errors": [],
            "release_window": selection.report(),
        }
    )
    if minute_source_audits:
        result["minute_source_audits"] = minute_source_audits
        result["minute_source_warnings"] = minute_source_warnings
    return result


def _split_codes(value: str) -> list[str]:
    return sorted({item.strip() for item in value.split(",") if item.strip()})


def _require_symbol_coverage(specs: list[FetchSpec], rows: list[dict]) -> None:
    row_count_by_key = {str(row["unit_key"]): int(row.get("row_count") or 0) for row in rows}
    totals: dict[tuple[str, str], int] = {}
    for spec in specs:
        symbol = str(spec.params["ts_code"])
        key = (spec.dataset, symbol)
        totals[key] = totals.get(key, 0) + row_count_by_key.get(spec.unit_key, 0)
    empty = [f"{dataset}/{symbol}" for (dataset, symbol), count in totals.items() if count == 0]
    if empty:
        raise RuntimeError(
            "minute download produced no rows for requested symbols: " + ", ".join(empty)
        )


def _write_optional_result(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _produce_factors(label: str, produce: Callable[[], Any]) -> Any:
    """Run a factor-production processor with a guaranteed non-zero failure exit.

    Fail-closed builders raise RuntimeError/ValueError on incomplete inputs
    (short trade calendar, missing parquets). Letting that propagate as an
    unhandled exception relies on the interpreter's excepthook for a non-zero
    exit code, which invocation wrappers (docker exec pipelines, shell
    capture) have flattened to 0 in production — schedulers then misread a
    failed run as success. Convert expected data errors to typer.Exit(2),
    matching the operational-failure convention used by bootstrap; usage
    errors stay BadParameter, and unexpected exceptions still propagate.
    """

    try:
        return produce()
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{label} failed (fail-closed)[/red]: {exc}")
        raise typer.Exit(2) from exc


def _execution_universe_contract(
    *,
    profile: str,
    symbols_by_dataset: dict[str, list[str]],
    universe_evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    """Seal the immutable universe definition used for compatible data rolls.

    Exact work-unit content and the requested end date belong to the dataset
    identity, not its stable lineage.  The lineage must nevertheless separate
    unrelated manual/auto universes; otherwise ``latest_compatible`` could roll
    an approved simulation onto minute data for a different set of securities.
    The full historical A-share feed is policy-defined, so newly listed symbols
    remain compatible additions to that one universe rather than creating a new
    lineage every day.
    """

    normalized_symbols = {
        str(dataset): sorted({str(symbol) for symbol in symbols if str(symbol)})
        for dataset, symbols in sorted(symbols_by_dataset.items())
    }
    evidence = dict(universe_evidence or {})
    if (
        profile == "ashare_intraday"
        and evidence.get("mode") == "historically_active_a_share_master"
        and evidence.get("source") == "stock_basic"
    ):
        payload: dict[str, Any] = {
            "mode": "historically_active_a_share_master",
            "source": "stock_basic",
        }
    else:
        payload = {
            "mode": "resolved_symbols",
            "symbols_by_dataset": normalized_symbols,
        }
    return {
        "version": "execution-universe-contract-v1",
        "payload": payload,
        "sha256": canonical_sha256(payload),
    }


def _build_execution_snapshot(
    context: Context,
    *,
    name: str,
    selected: dict[str, list[dict]],
    start_date: date,
    end_date: date,
    symbols_by_dataset: dict[str, list[str]],
    universe_evidence: dict | None = None,
    frequency: str = "1min",
    profile: str = "pair_execution",
    source_lineage_id: str | None = None,
    source_lineage_evidence: dict[str, Any] | None = None,
    quality_gate: dict[str, Any] | None = None,
) -> Path:
    module_root = Path(__file__).resolve().parent
    universe_contract = _execution_universe_contract(
        profile=profile,
        symbols_by_dataset=symbols_by_dataset,
        universe_evidence=universe_evidence,
    )
    lineage_configuration = {
        "contract_version": EXECUTION_SNAPSHOT_CONTRACT_VERSION,
        "start_date": start_date.isoformat(),
        "frequency": frequency,
        "provider": "tushare-compatible",
        "universe_contract_sha256": universe_contract["sha256"],
        "ingestion_contract_sha256": file_contract_sha256(
            {
                "execution_data": module_root / "execution_data.py",
                "provider": module_root / "provider.py",
                "storage": module_root / "storage.py",
            }
        ),
    }
    paired_source_lineage_id = str(source_lineage_id or "")
    if len(paired_source_lineage_id) != 64 or any(
        character not in "0123456789abcdef" for character in paired_source_lineage_id
    ):
        raise ValueError("execution snapshot source lineage must be a SHA-256 digest")
    if (
        not isinstance(source_lineage_evidence, dict)
        or source_lineage_evidence.get("source_lineage_id") != paired_source_lineage_id
        or not _is_sha256(source_lineage_evidence.get("evidence_sha256"))
        or not _is_sha256(source_lineage_evidence.get("qlib_dataset_lineage_id"))
    ):
        raise ValueError("execution snapshot requires verified daily-source evidence")
    evidence_payload = {
        key: value
        for key, value in source_lineage_evidence.items()
        if key != "evidence_sha256"
    }
    expected_evidence_sha256 = hashlib.sha256(
        json.dumps(evidence_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if source_lineage_evidence["evidence_sha256"] != expected_evidence_sha256:
        raise ValueError("execution snapshot daily-source evidence digest is inconsistent")
    if not isinstance(quality_gate, dict) or quality_gate.get("ok") is not True:
        raise ValueError("execution snapshot requires a passing scoped quality gate")
    lineage_configuration.update(
        {
            "source_lineage_id": paired_source_lineage_id,
            "source_qlib_dataset_lineage_id": source_lineage_evidence[
                "qlib_dataset_lineage_id"
            ],
        }
    )
    lineage_contract = {"kind": profile, "configuration": lineage_configuration}
    lineage_id = make_lineage_id(profile, lineage_configuration)
    expected = {
        "profile": profile,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "frequency": frequency,
        "symbols": symbols_by_dataset,
        "universe": universe_evidence or {"mode": "manual"},
        "universe_contract": universe_contract,
        "lineage_id": lineage_id,
        "source_lineage_id": paired_source_lineage_id,
        "source_lineage_evidence": source_lineage_evidence,
        "selected_unit_set_sha256": quality_gate.get("selected_unit_set_sha256"),
        "plan_scope_sha256": quality_gate.get("plan_scope_sha256"),
    }
    existing = context.storage.snapshots_root / name
    if existing.exists():
        try:
            manifest = json.loads((existing / "manifest.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise ValueError(f"existing execution snapshot {name!r} is incomplete") from exc
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError(f"existing execution snapshot {name!r} has different inputs")
        if set(manifest.get("datasets", {})) != set(selected):
            raise ValueError(f"existing execution snapshot {name!r} has different datasets")
        existing_gate = manifest.get("quality_gate")
        stable_gate_fields = (
            "ok",
            "plan_scope_sha256",
            "selected_unit_set_sha256",
            "daily_source_evidence_sha256",
        )
        if not isinstance(existing_gate, dict) or any(
            existing_gate.get(key) != quality_gate.get(key)
            for key in stable_gate_fields
        ):
            raise ValueError(
                f"existing execution snapshot {name!r} has different quality evidence"
            )
        verified_manifest = verify_snapshot_lineage(existing)
        if verified_manifest != manifest:
            raise ValueError(f"existing execution snapshot {name!r} lineage changed")
        QlibBuilder(existing)._snapshot_manifest_digest()
        return existing
    return context.storage.build_snapshot(
        name=name,
        successful_units=selected,
        manifest_extra={
            **expected,
            "quality_gate": quality_gate,
            "parent_snapshot": None,
            "parent_manifest_sha256": None,
            "lineage_generation": 0,
            "provider": "tushare-compatible",
            "lineage_contract": lineage_contract,
        },
    )


def _profile_datasets(profile: str) -> set[str]:
    if profile == RESEARCH_ASSET_SNAPSHOT_PROFILE:
        return {"trade_cal", "research_report"}
    if profile == GLOBAL_REFERENCE_SNAPSHOT_PROFILE:
        return set(GLOBAL_REFERENCE_DATASETS)
    datasets = {
        "stock_basic",
        "trade_cal",
        "index_basic",
        "index_daily",
        "index_dailybasic",
        "index_weight",
        *(item.name for item in CORE_DAILY),
    }
    if profile in {"research", "full"}:
        datasets.update(item.name for item in (*RESEARCH_DAILY, *ETF_DAILY))
        datasets.update(
            {
                "fund_basic",
                "index_classify",
                "index_member_all",
                "disclosure_date",
            }
        )
    if profile == "full":
        datasets.update(item.name for item in (*FUNDAMENTALS, *CORPORATE_EVENTS))
        datasets.add("news")
    return datasets


def _snapshot_datasets(profile: str, *, available: set[str]) -> set[str]:
    """Return the exact checkpoint datasets one profile will publish.

    Strict verification is scoped to this set so unrelated long-running jobs
    cannot block a core/research snapshot.  Full snapshots remain strict over
    every non-minute dataset they would actually include.
    """

    if profile not in SNAPSHOT_PROFILES:
        raise typer.BadParameter(
            "profile must be core, research, full, research-assets, or "
            "global-reference"
        )
    datasets = _profile_datasets(profile) | set(_required_profile_datasets(profile))
    # The profile is an explicit publication contract. Unrelated supplemental,
    # text or minute downloads in the shared checkpoint must not silently join
    # it or block a governed Qlib release merely because they already exist.
    return datasets


def _required_profile_datasets(profile: str) -> frozenset[str]:
    """Datasets that must exist before one profile can be published."""

    if profile == RESEARCH_ASSET_SNAPSHOT_PROFILE:
        return frozenset({"trade_cal", "research_report"})
    if profile == GLOBAL_REFERENCE_SNAPSHOT_PROFILE:
        # The profile is an explicit publication contract: every registered
        # peripheral dataset must be present before the snapshot publishes.
        return GLOBAL_REFERENCE_DATASETS
    return (
        QLIB_RESEARCH_REQUIRED_DATASETS
        if profile == "full"
        else QLIB_DAILY_REQUIRED_DATASETS
    )


if __name__ == "__main__":
    app()
