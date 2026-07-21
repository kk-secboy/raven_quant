from __future__ import annotations

import json
import threading
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from quant_platform.announcement_nlp import process_announcements
from quant_platform.corpus_nlp import SUPPORTED_CORPUS_DATASETS, process_corpus
from quant_platform.major_news_mentions import process_major_news_mentions
from quant_platform.news_flash_factors import process_news_flash
from quant_platform.report_rc_factors import process_report_rc
from quant_platform.runtime_secret_store import RuntimeSecretStore

from .catalog import (
    CORE_DAILY,
    CORPORATE_EVENTS,
    ETF_DAILY,
    FUNDAMENTALS,
    REFERENCE_FIELDS,
    RESEARCH_DAILY,
)
from .checkpoint import CheckpointStore
from .cninfo_announcements import download_cninfo_announcements
from .config import Settings
from .coverage_data import coverage_secondary_specs
from .execution_data import MARGIN_DATASET, MINUTE_DATASETS, margin_specs
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
from .qlib_builder import QlibBuilder
from .rate_limit import GlobalRateGate
from .reference_data import select_current_reference_units
from .runner import DownloadRunner
from .snapshot_lineage import file_contract_sha256, make_lineage_id, prepare_lineage_metadata
from .storage import ParquetStore
from .supplemental_data import (
    SUPPORTED_BUNDLES,
    a_share_bulk_history_specs,
    bond_reference_specs,
    etf_constituent_history_specs,
    etf_constituent_overflow_repartition_specs,
    market_daily_specs,
    market_financial_specs,
    next_pagination_specs,
    require_pagination_terminated,
    share_float_overflow_repartition_specs,
    supplemental_specs,
)
from .universe import select_intraday_universe_from_store
from .verify import quality_gate_payload, verify_downloads, write_report

app = typer.Typer(no_args_is_help=True, help="Resumable Tushare-to-Parquet bootstrap pipeline")
console = Console()


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
    if "overflow continuation" in normalized or "partition continuation" in normalized:
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
    inserted = context.checkpoint.add(specs)
    datasets = {spec.dataset for spec in specs}
    ignored_keys: set[str] = set()
    recovery_specs, recovered_keys = _pagination_overflow_recovery(
        context, specs, ignored_keys
    )
    if recovered_keys:
        ignored_keys.update(recovered_keys)
        known = {spec.unit_key for spec in specs}
        recovery_specs = [spec for spec in recovery_specs if spec.unit_key not in known]
        specs.extend(recovery_specs)
        inserted += context.checkpoint.add(recovery_specs)
    _run_phase(context, label, datasets)

    while True:
        recovery_specs, recovered_keys = _pagination_overflow_recovery(
            context, specs, ignored_keys
        )
        if recovered_keys:
            ignored_keys.update(recovered_keys)
            known = {spec.unit_key for spec in specs}
            recovery_specs = [spec for spec in recovery_specs if spec.unit_key not in known]
            specs.extend(recovery_specs)
            inserted += context.checkpoint.add(recovery_specs)
            _run_phase(context, f"{label} overflow continuation", datasets)
            continue

        active_specs = [spec for spec in specs if spec.unit_key not in ignored_keys]
        rows = _require_specs_complete(context, active_specs)
        next_specs = next_pagination_specs(active_specs, rows)
        if not next_specs:
            recovery_specs, recovered_keys = _full_page_partition_recovery(
                context, active_specs, rows, ignored_keys
            )
            if recovered_keys:
                known = {spec.unit_key for spec in specs}
                recovery_specs = [spec for spec in recovery_specs if spec.unit_key not in known]
                specs.extend(recovery_specs)
                inserted += context.checkpoint.add(recovery_specs)
                _run_phase(context, f"{label} partition continuation", datasets)
                continue
            require_pagination_terminated(active_specs, rows)
            context.report_progress(
                "verifying", f"{label} pagination verified", datasets, force=True
            )
            return specs, rows, inserted

        specs.extend(next_specs)
        recovery_specs, recovered_keys = _pagination_overflow_recovery(
            context, specs, ignored_keys
        )
        if recovered_keys:
            ignored_keys.update(recovered_keys)
            known = {spec.unit_key for spec in specs}
            recovery_specs = [spec for spec in recovery_specs if spec.unit_key not in known]
            specs.extend(recovery_specs)
            inserted += context.checkpoint.add(recovery_specs)
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
}


def _reconcile_range_plan(context: Context, specs: list[FetchSpec]) -> list[FetchSpec]:
    """Reuse complete legacy partitions and plan only uncovered session gaps."""

    targets = [
        spec
        for spec in specs
        if spec.dataset in _RANGE_REUSE_DATASETS and is_adaptive_partition(spec)
    ]
    if not targets:
        return specs
    untouched = [spec for spec in specs if spec not in targets]
    rows_by_dataset = {
        dataset: context.checkpoint.successful(dataset)
        for dataset in {spec.dataset for spec in targets}
    }
    success_index: dict[
        str, dict[tuple[tuple[str, object], ...], list[FetchSpec]]
    ] = {}
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
            covered.update(
                value for value in values if candidate_start <= value <= candidate_end
            )
        segment: list[date] = []
        for value in values:
            if value in covered:
                if segment:
                    replacements.append(
                        resize_partition_spec(target, segment[0], segment[-1])
                    )
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
            if spec.unit_key not in planned_keys and not is_adaptive_partition(spec):
                stale.append(spec.unit_key)
    context.checkpoint.supersede_units(
        stale,
        "legacy unfinished unit superseded by adaptive range planning",
    )
    return planned


def _complete_success_specs(rows: list[dict]) -> list[FetchSpec]:
    specs_by_group: dict[str, list[tuple[FetchSpec, int]]] = {}
    result: list[FetchSpec] = []
    for row in rows:
        spec = _checkpoint_row_spec(row)
        group = spec.scope.get("page_group")
        if not group:
            result.append(spec)
            continue
        specs_by_group.setdefault(str(group), []).append(
            (spec, int(row.get("row_count") or 0))
        )
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
    """Replace provider-capped monthly pages with disjoint tail continuations."""

    rows_by_key = {
        str(row["unit_key"]): row
        for row in context.checkpoint.unit_rows(spec.unit_key for spec in specs)
    }
    recovery_specs: list[FetchSpec] = []
    recovered_keys: set[str] = set()
    for failed_spec in specs:
        if failed_spec.unit_key in ignored_keys or failed_spec.dataset != "share_float":
            continue
        row = rows_by_key.get(failed_spec.unit_key)
        if not row or str(row.get("status")) not in {"failed", "superseded"}:
            continue
        error = str(row.get("last_error") or "")
        normalized_error = error.replace("-", " ")
        offset = int(failed_spec.params.get("offset") or 0)
        if offset < 100_000 or (
            "code=50101" not in error and "offset cap" not in normalized_error
        ):
            continue

        recovery_specs.extend(share_float_overflow_repartition_specs(failed_spec))
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

    recovery_specs, recovered_keys = _share_float_overflow_recovery(
        context, specs, ignored_keys
    )
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
        if offset < 100_000 or (
            "code=50101" not in error and "offset cap" not in normalized_error
        ):
            continue

        if (
            failed_spec.params.get("ts_code")
            and failed_spec.params.get("start_date")
            and failed_spec.params.get("end_date")
        ):
            recovery_specs.extend(
                etf_constituent_overflow_repartition_specs(failed_spec)
            )
        else:
            if etf_master is None:
                etf_master = context.storage.read_units(
                    context.checkpoint.successful("etf_basic")
                )
                if "ts_code" not in etf_master.columns:
                    raise RuntimeError(
                        "etf_basic did not provide ts_code for constituent overflow recovery"
                    )
            symbols = _eligible_etf_symbols(
                etf_master,
                dataset=failed_spec.dataset,
                trade_date=str(failed_spec.params["trade_date"]),
            )
            recovery_specs.extend(
                etf_constituent_overflow_repartition_specs(failed_spec, symbols)
            )
        recovered_keys.add(failed_spec.unit_key)
        context.checkpoint.supersede_units(
            [failed_spec.unit_key],
            f"{error}; pagination offset cap superseded by smaller ETF partitions",
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
    continued = {
        str(parent)
        for spec in specs
        if (
            parent := spec.scope.get("continues_page_group")
            or spec.scope.get("supersedes_page_group")
        )
    }
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
        if not is_adaptive_partition(current):
            continue
        page_size = int(current.scope["page_size"])
        max_pages = int(current.scope["max_pages"])
        page_index = int(
            current.scope.get(
                "page_index", int(current.scope.get("offset") or 0) // page_size
            )
        )
        if page_index + 1 < max_pages or row_counts.get(current.unit_key, -1) < page_size:
            continue
        children.extend(split_partition_spec(current))
        recovered.add(current.unit_key)
        context.checkpoint.supersede_units(
            [page.unit_key for page in pages],
            "full final pagination page superseded by disjoint adaptive child partitions",
        )
    return children, recovered


def _eligible_etf_symbols(
    master: pd.DataFrame, *, dataset: str, trade_date: str
) -> list[str]:
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
            (min(previous[0], clipped[0]), max(previous[1], clipped[1]))
            if previous
            else clipped
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
    for row in context.checkpoint.unfinished_units(
        {"etf_sh_cons", "etf_sz_cons"}
    ):
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
        console.print(
            f"retired legacy per-ETF/per-day constituent units: {superseded}"
        )

    return [
        spec
        for spec in base_specs
        if spec.dataset not in {"etf_sh_cons", "etf_sz_cons"}
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
    profile: Annotated[str, typer.Option(help="core, research, or full")] = "core",
    start: Annotated[str, typer.Option(help="YYYY-MM-DD")] = "2024-01-01",
    end: Annotated[str, typer.Option(help="YYYY-MM-DD or latest")] = "latest",
    snapshot_name: Annotated[str | None, typer.Option("--snapshot-name")] = None,
    build_qlib: Annotated[
        bool, typer.Option("--build-qlib/--no-build-qlib", help="Build Qlib .bin after snapshot")
    ] = True,
    download_only: Annotated[
        bool,
        typer.Option("--download-only", help="Stop after durable download units complete"),
    ] = False,
) -> None:
    """Plan, download, verify, and snapshot an initialization range."""
    if profile not in {"core", "research", "full"}:
        raise typer.BadParameter("profile must be core, research, or full")
    start_date = parse_date(start)
    end_date = parse_date(end, latest=today_cn())
    if end_date < start_date:
        raise typer.BadParameter("end must not be before start")
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
    reference_failures = [
        row
        for row in context.checkpoint.failures(1000)
        if row["dataset"] in {"stock_basic", "trade_cal", "index_basic"}
    ]
    if reference_failures:
        console.print("[red]reference phase failed; daily planning was not attempted[/red]")
        raise typer.Exit(2)

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
        planned_members = context.planner.plan_industry_members(
            max_attempts, as_of=end_date
        )
        console.print(f"planned historical industry membership units: +{planned_members}")
        _run_phase(context, "historical industry members", {"index_member_all"})
    if profile == "full":
        bulk_specs = a_share_bulk_history_specs(
            start=start_date,
            end=end_date,
            max_attempts=max_attempts,
        )
        _, _, bulk_inserted = _run_paginated_specs(
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

    if download_only:
        console.print("[bold green]download phase complete[/bold green]")
        return

    report = verify_downloads(
        context.checkpoint,
        context.settings.data_root,
        snapshot_end=end_date,
        require_all_planned=False,
    )
    report_path = context.settings.data_root / "verification" / "latest.json"
    write_report(report, report_path)
    if not report["ok"]:
        console.print(f"[red]verification failed[/red]: {report_path}")
        for error in report["errors"][:20]:
            console.print(f"  - {error}")
        raise typer.Exit(3)

    name = snapshot_name or (
        f"cn-{start_date:%Y%m%d}-{end_date:%Y%m%d}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    )
    snapshot_path = _build_snapshot(
        context, name, start_date, end_date, profile, quality_gate=quality_gate_payload(report)
    )
    write_report(report, snapshot_path / "verification.json")
    if build_qlib:
        qlib_path = _build_qlib(context, snapshot_path, staging_only=False)
        console.print(f"[green]Qlib dataset built[/green]: {qlib_path}")
    console.print(f"[bold green]bootstrap complete[/bold green]: {snapshot_path}")


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


@app.command()
def verify(
    snapshot_end: Annotated[
        str, typer.Option("--snapshot-end", help="Successor snapshot end date")
    ] = "latest",
    allow_incomplete_plans: Annotated[
        bool,
        typer.Option(
            "--allow-incomplete-plans",
            help="Warn about unrelated dormant plans instead of failing the pipeline",
        ),
    ] = False,
) -> None:
    """Validate checkpoints, files, checksums, empties, and duplicate core keys."""
    context = load_context(require_credentials=False)
    report = verify_downloads(
        context.checkpoint,
        context.settings.data_root,
        snapshot_end=parse_date(snapshot_end, latest=today_cn()),
        require_all_planned=not allow_incomplete_plans,
    )
    path = context.settings.data_root / "verification" / "latest.json"
    write_report(report, path)
    console.print_json(json.dumps(report, ensure_ascii=False))
    if not report["ok"]:
        raise typer.Exit(3)


@app.command()
def snapshot(
    name: Annotated[str | None, typer.Option()] = None,
    start: Annotated[str, typer.Option()] = "2024-01-01",
    end: Annotated[str, typer.Option()] = "latest",
    profile: Annotated[str, typer.Option()] = "core",
) -> None:
    """Build an immutable compacted Parquet snapshot from successful units."""
    context = load_context(require_credentials=False)
    start_date = parse_date(start)
    end_date = parse_date(end, latest=today_cn())
    name = name or f"cn-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    # Verify before building so every snapshot manifest records an explicit
    # quality gate; Qlib builds refuse snapshots without quality_gate.ok=true.
    report = verify_downloads(
        context.checkpoint,
        context.settings.data_root,
        snapshot_end=end_date,
        require_all_planned=False,
    )
    write_report(report, context.settings.data_root / "verification" / "latest.json")
    if not report["ok"]:
        console.print("[red]verification failed; snapshot records quality_gate.ok=false[/red]")
        for error in report["errors"][:20]:
            console.print(f"  - {error}")
    path = _build_snapshot(
        context, name, start_date, end_date, profile, quality_gate=quality_gate_payload(report)
    )
    write_report(report, path / "verification.json")
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
    result_path: Annotated[Path | None, typer.Option("--result")] = None,
) -> None:
    """Download bounded 1-minute windows and build pair-execution evidence."""
    start_date = parse_date(start)
    end_date = parse_date(end, latest=today_cn())
    if end_date < start_date:
        raise typer.BadParameter("end must not be before start")
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
    universe_evidence: dict | None = None
    if auto_universe:
        selected = select_intraday_universe_from_store(
            context.checkpoint,
            context.storage,
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
        universe_evidence = selected.evidence
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
    trading_dates = context.planner.trading_dates(start_date, end_date)
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
    selected: dict[str, list[dict]] = {MARGIN_DATASET: margin_rows}
    for dataset in sorted(minute_datasets):
        keys = {spec.unit_key for spec in specs if spec.dataset == dataset}
        selected[dataset] = [row for row in minute_rows if row["unit_key"] in keys]
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
    result_path: Annotated[Path | None, typer.Option("--result")] = None,
) -> None:
    """Download resumable 5-minute bars for every A-share active in the range."""

    start_date = parse_date(start)
    end_date = parse_date(end, latest=today_cn())
    if end_date < start_date:
        raise typer.BadParameter("end must not be before start")
    context = load_context(
        progress_path=result_path,
        progress_target={
            "kind": "ashare_5m_download",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "frequency": "5min",
        },
    )
    master = context.storage.read_units(context.checkpoint.successful("stock_basic"))
    active_ranges = _historical_a_share_active_ranges(
        master, start=start_date, end=end_date
    )
    symbols = sorted(active_ranges)

    if not symbols:
        raise RuntimeError("stock_basic produced an empty historical A-share universe")
    context.progress.set_target(symbols=len(symbols))
    context.report_progress(
        "planning", "full A-share 5-minute planning", {"ashare_5m"}, force=True
    )
    specs = context.execution_planner.plan_minutes(
        {"ashare_5m": symbols},
        start_date,
        end_date,
        context.settings.max_request_attempts,
        freq="5min",
        active_ranges_by_dataset={"ashare_5m": active_ranges},
        trading_dates=context.planner.trading_dates(start_date, end_date),
    )
    specs, rows, _ = _run_paginated_specs(
        context, "full A-share 5-minute bars", specs
    )
    _require_symbol_coverage(specs, rows)
    name = snapshot_name or (
        f"ashare-5m-{start_date:%Y%m%d}-{end_date:%Y%m%d}-"
        f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    )
    context.report_progress(
        "snapshot", "building immutable 5-minute snapshot", {"ashare_5m"}, force=True
    )
    snapshot_path = _build_execution_snapshot(
        context,
        name=name,
        selected={"ashare_5m": rows},
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
    )
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
    ts_code: Annotated[
        str, typer.Option(help="Comma-separated Tushare codes to include")
    ] = "",
    start: Annotated[str, typer.Option(help="YYYY-MM-DD announcement date")] = "2024-01-01",
    end: Annotated[str, typer.Option(help="YYYY-MM-DD or latest")] = "latest",
    limit: Annotated[
        int, typer.Option(min=0, help="Maximum announcements to download (0 = all)")
    ] = 0,
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
    summary = download_cninfo_announcements(
        context.settings.data_root,
        ts_codes=set(_split_codes(ts_code)) or None,
        start=start_date,
        end=end_date,
        limit=limit or None,
        rate_gate=context.rate_gate,
        timeout_seconds=context.settings.timeout_seconds,
        max_attempts=context.settings.max_request_attempts,
        cooldown_seconds=context.settings.cooldown_seconds,
    )
    result = {
        "dataset": "cninfo_announcements",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "ts_codes": _split_codes(ts_code),
        **summary.as_dict(),
    }
    _write_optional_result(result_path, result)
    console.print_json(json.dumps(result, ensure_ascii=False))


@app.command("announcement-nlp")
def announcement_nlp_command(
    ts_code: Annotated[
        str, typer.Option(help="Comma-separated Tushare codes to include")
    ] = "",
    start: Annotated[str, typer.Option(help="YYYY-MM-DD announcement date")] = "2024-01-01",
    end: Annotated[str, typer.Option(help="YYYY-MM-DD or latest")] = "latest",
    category: Annotated[
        str,
        typer.Option(help="Comma-separated announcement|regulatory_letter; empty = all"),
    ] = "",
    limit: Annotated[
        int, typer.Option(min=0, help="Maximum announcements to process (0 = all)")
    ] = 0,
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
    settings = context.settings
    secret_store = RuntimeSecretStore(settings.database_url, settings.platform_secret_key)
    summary = process_announcements(
        settings.data_root,
        ts_codes=set(_split_codes(ts_code)) or None,
        start=start_date,
        end=end_date,
        categories=categories or None,
        limit=limit or None,
        secret_store=secret_store,
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


@app.command("corpus-nlp")
def corpus_nlp_command(
    dataset: Annotated[
        str,
        typer.Option(
            help="Comma-separated major_news,npr,cctv_news,irm_qa_sh,irm_qa_sz; "
            "empty = all"
        ),
    ] = "",
    ts_code: Annotated[
        str, typer.Option(help="Comma-separated Tushare codes to include")
    ] = "",
    start: Annotated[str, typer.Option(help="YYYY-MM-DD publication date")] = "2024-01-01",
    end: Annotated[str, typer.Option(help="YYYY-MM-DD or latest")] = "latest",
    limit: Annotated[
        int, typer.Option(min=0, help="Maximum corpus items to process (0 = all)")
    ] = 0,
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
    context.report_progress(
        "processing", "corpus NLP extraction", {"corpus_nlp"}, force=True
    )
    settings = context.settings
    secret_store = RuntimeSecretStore(settings.database_url, settings.platform_secret_key)
    summary = process_corpus(
        settings.data_root,
        datasets=datasets or None,
        ts_codes=set(_split_codes(ts_code)) or None,
        start=start_date,
        end=end_date,
        limit=limit or None,
        secret_store=secret_store,
    )
    result = {
        "dataset": "corpus_nlp",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "datasets": sorted(datasets) or list(SUPPORTED_CORPUS_DATASETS),
        "ts_codes": _split_codes(ts_code),
        **summary.as_dict(),
    }
    _write_optional_result(result_path, result)
    console.print_json(json.dumps(result, ensure_ascii=False))


@app.command("report-rc-factors")
def report_rc_factors_command(
    ts_code: Annotated[
        str, typer.Option(help="Comma-separated Tushare codes to include")
    ] = "",
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
    summary = process_report_rc(
        context.settings.data_root,
        ts_codes=set(_split_codes(ts_code)) or None,
        start=start_date,
        end=end_date,
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


@app.command("major-news-mentions")
def major_news_mentions_command(
    ts_code: Annotated[
        str, typer.Option(help="Comma-separated Tushare codes to include")
    ] = "",
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
    context.report_progress(
        "processing", "major_news mention mapping", {"major_news"}, force=True
    )
    summary = process_major_news_mentions(
        context.settings.data_root,
        ts_codes=set(_split_codes(ts_code)) or None,
        start=start_date,
        end=end_date,
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
    result_path: Annotated[Path | None, typer.Option("--result")] = None,
) -> None:
    """Build the market-level news-flash intensity factor artifact."""
    context = load_context(
        require_credentials=False,
        progress_path=result_path,
        progress_target={"kind": "news_flash_factors"},
    )
    context.report_progress(
        "processing", "news flash intensity factor production", {"news"}, force=True
    )
    summary = process_news_flash(context.settings.data_root)
    result = {"dataset": "news", **summary.as_dict()}
    _write_optional_result(result_path, result)
    console.print_json(json.dumps(result, ensure_ascii=False))


def _historical_a_share_symbols(
    master: pd.DataFrame, *, start: date, end: date
) -> list[str]:
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
        & master["ts_code"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.endswith((".SH", ".SZ", ".BJ"))
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
            min(previous[0], clipped[0]),
            max(previous[1], clipped[1]),
        ) if previous else clipped
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


def _open_market_dates(
    calendar: pd.DataFrame, *, start: date, end: date
) -> list[str]:
    if calendar.empty or "is_open" not in calendar.columns:
        raise RuntimeError("market trade calendar is unavailable or missing is_open")
    date_field = "cal_date" if "cal_date" in calendar.columns else "date"
    if date_field not in calendar.columns:
        raise RuntimeError("market trade calendar is missing cal_date/date")
    values = pd.to_datetime(calendar[date_field], errors="coerce")
    is_open = calendar["is_open"].astype(str).str.lower().isin({"1", "true", "t", "yes"})
    selected = is_open & values.between(pd.Timestamp(start), pd.Timestamp(end))
    return sorted(values.loc[selected].dt.strftime("%Y%m%d").dropna().unique().tolist())


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
    if specs:
        specs, rows, inserted = _run_paginated_specs(context, bundle, specs)
    console.print(
        f"planned supplemental bundle={bundle} units={len(specs)} newly_inserted={inserted}"
    )
    if bundle == "cn_governance_risk":
        master = context.storage.read_units(context.checkpoint.successful("stock_basic"))
        if "ts_code" not in master.columns:
            raise RuntimeError(
                "stock_basic did not provide ts_code for management-reward planning"
            )
        secondary_specs = coverage_secondary_specs(
            bundle,
            {
                "stk_rewards": sorted(
                    {
                        str(value).strip()
                        for value in master["ts_code"].dropna().tolist()
                        if str(value).strip()
                    }
                )
            },
            start=start_date,
            end=end_date,
            max_attempts=context.settings.max_request_attempts,
        )
        inserted += context.checkpoint.add(secondary_specs)
        context.checkpoint.retry_failed_units(spec.unit_key for spec in secondary_specs)
        secondary_datasets = {spec.dataset for spec in secondary_specs}
        _run_phase(context, f"{bundle} symbol batches", secondary_datasets)
        secondary_rows = _require_specs_complete(context, secondary_specs)
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
        calendar = context.storage.read_units(
            context.checkpoint.successful(calendar_dataset)
        )
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
            help="Build even when the snapshot manifest has no passing quality gate",
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
    target_frequency: Annotated[
        str | None, typer.Option("--target-frequency")
    ] = None,
    staging_only: Annotated[bool, typer.Option("--staging-only")] = False,
    skip_quality_gate: Annotated[
        bool,
        typer.Option(
            "--skip-quality-gate",
            help="Build even when the snapshot manifest has no passing quality gate",
        ),
    ] = False,
) -> None:
    """Build native or Qlib-resampled data without another download path."""
    context = load_context(require_credentials=False)
    snapshot_path = context.storage.snapshots_root / snapshot_name
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
            "and snapshot commands, or pass --skip-quality-gate to override"
        )


def _build_snapshot(
    context: Context,
    name: str,
    start_date: date,
    end_date: date,
    profile: str,
    quality_gate: dict[str, Any] | None = None,
) -> Path:
    module_root = Path(__file__).resolve().parent
    lineage_id = make_lineage_id(
        "qlib_daily_source",
        {
            "profile": profile,
            "start_date": start_date.isoformat(),
            "provider": "tushare-compatible",
            "ingestion_contract_sha256": file_contract_sha256(
                {
                    "planner": module_root / "planner.py",
                    "provider": module_root / "provider.py",
                    "storage": module_root / "storage.py",
                }
            ),
        },
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
        return existing
    available = set(context.checkpoint.datasets())
    selected_datasets = _profile_datasets(profile)
    if profile == "full":
        # A full immutable research lake must retain every downloaded daily,
        # reference and alternative dataset. Minute bars and shortability form
        # a separate execution snapshot with a different frequency contract.
        selected_datasets.update(available - set(MINUTE_DATASETS) - {MARGIN_DATASET})
    units = {
        dataset: select_current_reference_units(
            context.checkpoint.successful(dataset), snapshot_end=end_date
        )
        for dataset in sorted(selected_datasets & available)
    }
    if not units:
        raise ValueError(f"no successful {profile} datasets are available for snapshotting")
    lineage = prepare_lineage_metadata(
        context.storage.snapshots_root,
        lineage_id=lineage_id,
        end_date=end_date,
        successful_units=units,
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
            "provider": "tushare-compatible",
            **({"quality_gate": quality_gate} if quality_gate else {}),
            **lineage,
        },
        base_snapshot=base_snapshot,
    )


def _build_qlib(
    context: Context,
    snapshot_path: Path,
    *,
    staging_only: bool,
    skip_quality_gate: bool = False,
) -> Path:
    _require_snapshot_quality_gate(snapshot_path, skip=skip_quality_gate)
    builder = QlibBuilder(snapshot_path)
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
            return output
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
        max_workers=min(16, max(1, context.settings.workers * 2)),
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
            return output
        raise ValueError(
            f"existing minute Qlib output is incomplete and requires operator review: {output}"
        )
    native_staging = (
        staging.with_name(f"{staging.name}.native")
        if builder.requires_resampling
        else staging
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
) -> Path:
    module_root = Path(__file__).resolve().parent
    lineage_id = make_lineage_id(
        profile,
        {
            "start_date": start_date.isoformat(),
            "frequency": frequency,
            "provider": "tushare-compatible",
            "symbols": symbols_by_dataset,
            "ingestion_contract_sha256": file_contract_sha256(
                {
                    "execution_data": module_root / "execution_data.py",
                    "provider": module_root / "provider.py",
                    "storage": module_root / "storage.py",
                }
            ),
        },
    )
    expected = {
        "profile": profile,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "frequency": frequency,
        "symbols": symbols_by_dataset,
        "universe": universe_evidence or {"mode": "manual"},
        "lineage_id": lineage_id,
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
        return existing
    lineage = prepare_lineage_metadata(
        context.storage.snapshots_root,
        lineage_id=lineage_id,
        end_date=end_date,
        successful_units=selected,
    )
    return context.storage.build_snapshot(
        name=name,
        successful_units=selected,
        manifest_extra={**expected, **lineage, "provider": "tushare-compatible"},
    )


def _profile_datasets(profile: str) -> set[str]:
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


if __name__ == "__main__":
    app()
