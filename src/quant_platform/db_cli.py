from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from alembic import command
from alembic.config import Config

from quant_data.config import Settings
from quant_platform.announcement_factor_registry import (
    IMPORT_ACTOR,
    default_factors_dir,
    register_announcement_factor,
)
from quant_platform.announcement_nlp import FACTOR_NAME, LOGIC_FACTOR_NAME
from quant_platform.corpus_nlp import (
    CORPUS_FACTOR_NAMES,
    register_corpus_factor,
)
from quant_platform.corpus_nlp import (
    IMPORT_ACTOR as CORPUS_IMPORT_ACTOR,
)
from quant_platform.corpus_nlp import (
    default_factors_dir as corpus_factors_dir,
)
from quant_platform.major_news_mentions import (
    FACTOR_NAMES as MENTIONS_FACTOR_NAMES,
)
from quant_platform.major_news_mentions import (
    IMPORT_ACTOR as MENTIONS_IMPORT_ACTOR,
)
from quant_platform.major_news_mentions import (
    default_factors_dir as mentions_factors_dir,
)
from quant_platform.major_news_mentions import (
    register_major_news_mentions_factor,
)
from quant_platform.multiface_audit import (
    audit_multiface_readiness,
    write_multiface_report,
)
from quant_platform.news_flash_factors import (
    IMPORT_ACTOR as NEWS_FLASH_IMPORT_ACTOR,
)
from quant_platform.news_flash_factors import (
    default_factors_dir as news_flash_factors_dir,
)
from quant_platform.news_flash_factors import (
    register_news_flash_factor,
)
from quant_platform.report_rc_factors import (
    FACTOR_NAMES as REPORT_RC_FACTOR_NAMES,
)
from quant_platform.report_rc_factors import (
    IMPORT_ACTOR as REPORT_RC_IMPORT_ACTOR,
)
from quant_platform.report_rc_factors import (
    default_factors_dir as report_rc_factors_dir,
)
from quant_platform.report_rc_factors import (
    register_report_rc_factor,
)
from quant_platform.research_store import ResearchStore

app = typer.Typer(no_args_is_help=True, help="QuantLab PostgreSQL schema management")


def project_root() -> Path:
    current = Path.cwd().resolve()
    if (current / "alembic.ini").exists():
        return current
    return Path(__file__).resolve().parents[2]


def alembic_config(database_url: str) -> Config:
    root = project_root()
    config = Config(root / "alembic.ini")
    config.set_main_option("script_location", str(root / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    return config


def upgrade_database(database_url: str, revision: str = "head") -> None:
    command.upgrade(alembic_config(database_url), revision)


@app.command()
def upgrade(
    revision: Annotated[str, typer.Option(help="Alembic revision to apply")] = "head",
) -> None:
    """Apply pending PostgreSQL schema migrations."""
    settings = Settings.from_env(project_root() / ".env")
    upgrade_database(settings.database_url, revision)


@app.command()
def current() -> None:
    """Print the current PostgreSQL schema revision."""
    settings = Settings.from_env(project_root() / ".env")
    command.current(alembic_config(settings.database_url), verbose=True)


@app.command("register-announcement-factor")
def register_announcement_factor_command(
    factor_name: Annotated[
        str, typer.Option(help="Announcement NLP factor artifact name")
    ] = "all",
    actor: Annotated[
        str, typer.Option(help="Actor recorded on the research run and events")
    ] = IMPORT_ACTOR,
) -> None:
    """Register the announcement NLP factor artifact into factor_candidates (idempotent)."""
    settings = Settings.from_env(project_root() / ".env")
    names = (
        [FACTOR_NAME, LOGIC_FACTOR_NAME]
        if factor_name == "all"
        else [part.strip() for part in factor_name.split(",") if part.strip()]
    )
    store = ResearchStore(settings.database_url)
    factors_dir = default_factors_dir(settings.data_root)
    results = [
        register_announcement_factor(store, factors_dir, factor_name=name, actor=actor)
        for name in names
    ]
    typer.echo(json.dumps({"factors": results}, ensure_ascii=False, indent=2))


@app.command("register-report-rc-factor")
def register_report_rc_factor_command(
    factor_name: Annotated[
        str,
        typer.Option(
            help="report_rc factor artifact name, or 'all' for every produced factor"
        ),
    ] = "all",
    actor: Annotated[
        str, typer.Option(help="Actor recorded on the research run and events")
    ] = REPORT_RC_IMPORT_ACTOR,
) -> None:
    """Register report_rc factor artifacts into factor_candidates (idempotent)."""
    settings = Settings.from_env(project_root() / ".env")
    names = (
        list(REPORT_RC_FACTOR_NAMES)
        if factor_name == "all"
        else [part.strip() for part in factor_name.split(",") if part.strip()]
    )
    store = ResearchStore(settings.database_url)
    factors_dir = report_rc_factors_dir(settings.data_root)
    results = [
        register_report_rc_factor(store, factors_dir, factor_name=name, actor=actor)
        for name in names
    ]
    typer.echo(json.dumps({"factors": results}, ensure_ascii=False, indent=2))


@app.command("register-corpus-factor")
def register_corpus_factor_command(
    factor_name: Annotated[
        str,
        typer.Option(
            help="corpus NLP factor artifact name, or 'all' for every produced factor"
        ),
    ] = "all",
    actor: Annotated[
        str, typer.Option(help="Actor recorded on the research run and events")
    ] = CORPUS_IMPORT_ACTOR,
) -> None:
    """Register corpus NLP factor artifacts into factor_candidates (idempotent)."""
    settings = Settings.from_env(project_root() / ".env")
    names = (
        list(CORPUS_FACTOR_NAMES)
        if factor_name == "all"
        else [part.strip() for part in factor_name.split(",") if part.strip()]
    )
    store = ResearchStore(settings.database_url)
    factors_dir = corpus_factors_dir(settings.data_root)
    results = [
        register_corpus_factor(store, factors_dir, factor_name=name, actor=actor)
        for name in names
    ]
    typer.echo(json.dumps({"factors": results}, ensure_ascii=False, indent=2))


@app.command("register-major-news-mentions-factor")
def register_major_news_mentions_factor_command(
    factor_name: Annotated[
        str,
        typer.Option(
            help="major_news mention factor artifact name, or 'all' for every produced factor"
        ),
    ] = "all",
    actor: Annotated[
        str, typer.Option(help="Actor recorded on the research run and events")
    ] = MENTIONS_IMPORT_ACTOR,
) -> None:
    """Register major_news mention factor artifacts into factor_candidates (idempotent)."""
    settings = Settings.from_env(project_root() / ".env")
    names = (
        list(MENTIONS_FACTOR_NAMES)
        if factor_name == "all"
        else [part.strip() for part in factor_name.split(",") if part.strip()]
    )
    store = ResearchStore(settings.database_url)
    factors_dir = mentions_factors_dir(settings.data_root)
    results = [
        register_major_news_mentions_factor(store, factors_dir, factor_name=name, actor=actor)
        for name in names
    ]
    typer.echo(json.dumps({"factors": results}, ensure_ascii=False, indent=2))


@app.command("register-news-flash-factor")
def register_news_flash_factor_command(
    actor: Annotated[
        str, typer.Option(help="Actor recorded on the research run and events")
    ] = NEWS_FLASH_IMPORT_ACTOR,
) -> None:
    """Register the news flash intensity factor artifact into factor_candidates."""
    settings = Settings.from_env(project_root() / ".env")
    result = register_news_flash_factor(
        ResearchStore(settings.database_url),
        news_flash_factors_dir(settings.data_root),
        actor=actor,
    )
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("audit-multiface")
def audit_multiface_command(
    dataset: Annotated[
        str, typer.Option(help="Immutable Qlib dataset directory name")
    ],
    snapshot_name: Annotated[
        str | None,
        typer.Option("--snapshot", help="Snapshot name; defaults to Qlib provenance"),
    ] = None,
    result: Annotated[
        Path | None,
        typer.Option(help="Optional job result JSON path"),
    ] = None,
    require_ready: Annotated[
        bool,
        typer.Option(help="Exit non-zero when any governed face is not ready"),
    ] = False,
) -> None:
    """Audit Qlib fields, information factors, OOS evidence and label isolation."""

    settings = Settings.from_env(project_root() / ".env")
    report = audit_multiface_readiness(
        settings.data_root,
        dataset=dataset,
        snapshot_name=snapshot_name,
        ledger=ResearchStore(settings.database_url),
    )
    report_path = write_multiface_report(settings.data_root, report)
    output = {**report, "report_path": str(report_path)}
    if result is not None:
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text(
            json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    typer.echo(json.dumps(output, ensure_ascii=False, indent=2))
    if require_ready and not report["ok"]:
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
