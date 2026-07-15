from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from .catalog import (
    CORE_DAILY,
    CORPORATE_EVENTS,
    ETF_DAILY,
    FUNDAMENTALS,
    REFERENCE_FIELDS,
    RESEARCH_DAILY,
)
from .checkpoint import CheckpointStore
from .config import Settings
from .execution_data import MARGIN_DATASET, MINUTE_DATASETS, margin_specs
from .minute_qlib_builder import MinuteQlibBuilder
from .models import FetchSpec
from .planner import BootstrapPlanner, ExecutionDataPlanner, compact_date, parse_date
from .provider import TushareHttpProvider
from .qlib_builder import QlibBuilder
from .rate_limit import GlobalRateGate
from .runner import DownloadRunner
from .snapshot_lineage import file_contract_sha256, make_lineage_id, prepare_lineage_metadata
from .storage import ParquetStore
from .supplemental_data import (
    SUPPORTED_BUNDLES,
    a_share_bulk_history_specs,
    bond_reference_specs,
    market_financial_specs,
    next_pagination_specs,
    require_pagination_terminated,
    supplemental_specs,
)
from .universe import select_intraday_universe_from_store
from .verify import verify_downloads, write_report

app = typer.Typer(no_args_is_help=True, help="Resumable Tushare-to-Parquet bootstrap pipeline")
console = Console()


class Context:
    def __init__(self, settings: Settings, on_result=None) -> None:
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


def load_context(*, require_credentials: bool = True, on_result=None) -> Context:
    settings = Settings.from_env()
    if require_credentials:
        settings.require_credentials()
    settings.data_root.mkdir(parents=True, exist_ok=True)
    return Context(settings, on_result=on_result)


def _run_phase(context: Context, label: str, datasets: set[str]) -> None:
    total = context.checkpoint.remaining_count(datasets)
    if total == 0:
        console.print(f"[green]{label}: already complete[/green]")
        return
    with Progress(console=console) as progress:
        task = progress.add_task(label, total=total)

        def on_result(_dataset: str, _succeeded: bool, _rows: int) -> None:
            progress.advance(task)

        context.runner.on_result = on_result
        summary = context.runner.run(datasets)
    console.print(
        f"{label}: succeeded={summary.succeeded} failed={summary.failed} rows={summary.rows}"
    )


def _run_paginated_specs(
    context: Context, label: str, initial_specs: list[FetchSpec]
) -> tuple[list[FetchSpec], list[dict], int]:
    specs = list(initial_specs)
    inserted = context.checkpoint.add(specs)
    context.checkpoint.retry_failed_units(spec.unit_key for spec in specs)
    datasets = {spec.dataset for spec in specs}
    _run_phase(context, label, datasets)
    rows = _require_specs_complete(context, specs)
    while next_specs := next_pagination_specs(specs, rows):
        inserted += context.checkpoint.add(next_specs)
        context.checkpoint.retry_failed_units(spec.unit_key for spec in next_specs)
        specs.extend(next_specs)
        _run_phase(context, f"{label} pagination", datasets)
        rows = _require_specs_complete(context, specs)
    require_pagination_terminated(specs, rows)
    return specs, rows, inserted


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
    end_date = parse_date(end, latest=date.today())
    if end_date < start_date:
        raise typer.BadParameter("end must not be before start")
    context = load_context()
    max_attempts = context.settings.max_request_attempts

    planned_reference = context.planner.plan_reference(start_date, end_date, max_attempts)
    console.print(f"planned reference units: +{planned_reference}")
    _run_phase(context, "reference", {"stock_basic", "trade_cal"})
    reference_failures = [
        row
        for row in context.checkpoint.failures(1000)
        if row["dataset"] in {"stock_basic", "trade_cal"}
    ]
    if reference_failures:
        console.print("[red]reference phase failed; daily planning was not attempted[/red]")
        raise typer.Exit(2)

    planned = context.planner.plan_profile(profile, start_date, end_date, max_attempts)
    console.print(f"planned data units: {json.dumps(planned, ensure_ascii=False)}")
    daily_datasets = {definition.name for definition in CORE_DAILY}
    daily_datasets.update({"index_daily", "index_weight"})
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
        planned_members = context.planner.plan_industry_members(max_attempts)
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
        _run_phase(context, "market news", {"news"})

    if download_only:
        console.print("[bold green]download phase complete[/bold green]")
        return

    report = verify_downloads(context.checkpoint, context.settings.data_root)
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
    snapshot_path = _build_snapshot(context, name, start_date, end_date, profile)
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
def verify() -> None:
    """Validate checkpoints, files, checksums, empties, and duplicate core keys."""
    context = load_context(require_credentials=False)
    report = verify_downloads(context.checkpoint, context.settings.data_root)
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
    end_date = parse_date(end, latest=date.today())
    name = name or f"cn-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    path = _build_snapshot(context, name, start_date, end_date, profile)
    console.print(path)


@app.command("margin-eligibility")
def margin_eligibility(
    start: Annotated[str, typer.Option(help="YYYY-MM-DD")] = "2024-01-01",
    end: Annotated[str, typer.Option(help="YYYY-MM-DD or latest")] = "latest",
    result_path: Annotated[Path | None, typer.Option("--result")] = None,
) -> None:
    """Download daily full-market margin-eligible security evidence."""
    start_date = parse_date(start)
    end_date = parse_date(end, latest=date.today())
    if end_date < start_date:
        raise typer.BadParameter("end must not be before start")
    context = load_context()
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
    end_date = parse_date(end, latest=date.today())
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
    context = load_context()
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
    _run_phase(context, "core intraday", minute_datasets)
    minute_rows = _require_specs_complete(context, specs)
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
    end_date = parse_date(end, latest=date.today())
    if end_date < start_date:
        raise typer.BadParameter("end must not be before start")
    context = load_context()
    trading_dates = (
        context.planner.trading_dates(start_date, end_date)
        if bundle.startswith("cn_") and bundle != "cn_macro"
        else []
    )
    specs = supplemental_specs(
        bundle,
        start=start_date,
        end=end_date,
        trading_dates=trading_dates,
        max_attempts=context.settings.max_request_attempts,
    )
    inserted = context.checkpoint.add(specs)
    context.checkpoint.retry_failed_units(spec.unit_key for spec in specs)
    datasets = {spec.dataset for spec in specs}
    console.print(
        f"planned supplemental bundle={bundle} units={len(specs)} newly_inserted={inserted}"
    )
    _run_phase(context, bundle, datasets)
    rows = _require_specs_complete(context, specs)
    while next_specs := next_pagination_specs(specs, rows):
        inserted += context.checkpoint.add(next_specs)
        context.checkpoint.retry_failed_units(spec.unit_key for spec in next_specs)
        specs.extend(next_specs)
        _run_phase(context, f"{bundle} pagination", datasets)
        rows = _require_specs_complete(context, specs)
    require_pagination_terminated(specs, rows)
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
        financial_symbols = _split_codes(symbols)
        if not financial_symbols:
            master = context.storage.read_units(context.checkpoint.successful(basic_dataset))
            if "ts_code" not in master.columns:
                raise RuntimeError(
                    f"{basic_dataset} did not provide ts_code for financial planning"
                )
            financial_symbols = sorted(
                {
                    str(value).strip()
                    for value in master["ts_code"].dropna().tolist()
                    if str(value).strip()
                }
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
    result = _build_qlib(context, snapshot_path, staging_only=staging_only)
    console.print(result)


@app.command("build-minute-qlib")
def build_minute_qlib_command(
    snapshot_name: Annotated[str, typer.Option("--snapshot")],
    output_name: Annotated[str | None, typer.Option("--output-name")] = None,
    staging_only: Annotated[bool, typer.Option("--staging-only")] = False,
) -> None:
    """Build an independent Qlib 1-minute dataset from an execution snapshot."""
    context = load_context(require_credentials=False)
    snapshot_path = context.storage.snapshots_root / snapshot_name
    result = _build_minute_qlib(
        context,
        snapshot_path,
        output_name=output_name or f"{snapshot_name}-1min",
        staging_only=staging_only,
    )
    console.print(result)


def _build_snapshot(
    context: Context, name: str, start_date: date, end_date: date, profile: str
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
        dataset: context.checkpoint.successful(dataset)
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
    return context.storage.build_snapshot(
        name=name,
        successful_units=units,
        manifest_extra={
            "profile": profile,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "provider": "tushare-compatible",
            **lineage,
        },
    )


def _build_qlib(context: Context, snapshot_path: Path, *, staging_only: bool) -> Path:
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
    output_name: str,
    staging_only: bool,
) -> Path:
    builder = MinuteQlibBuilder(snapshot_path)
    staging = context.settings.data_root / "qlib_staging" / output_name
    output = context.settings.data_root / "qlib" / output_name
    if not staging_only and output.exists():
        required = (
            output / "calendars" / "1min.txt",
            output / "instruments" / "all.txt",
            output / "features",
            output / "metadata" / "provenance.json",
        )
        if all(path.exists() for path in required) and any(
            (output / "features").rglob("*.1min.bin")
        ):
            return output
        raise ValueError(
            f"existing minute Qlib output is incomplete and requires operator review: {output}"
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
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_execution_snapshot(
    context: Context,
    *,
    name: str,
    selected: dict[str, list[dict]],
    start_date: date,
    end_date: date,
    symbols_by_dataset: dict[str, list[str]],
    universe_evidence: dict | None = None,
) -> Path:
    module_root = Path(__file__).resolve().parent
    lineage_id = make_lineage_id(
        "pair_execution",
        {
            "start_date": start_date.isoformat(),
            "frequency": "1min",
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
        "profile": "pair_execution",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "frequency": "1min",
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
        "index_daily",
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
