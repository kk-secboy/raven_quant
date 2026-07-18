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
from quant_platform.announcement_nlp import FACTOR_NAME
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
    ] = FACTOR_NAME,
    actor: Annotated[
        str, typer.Option(help="Actor recorded on the research run and events")
    ] = IMPORT_ACTOR,
) -> None:
    """Register the announcement NLP factor artifact into factor_candidates (idempotent)."""
    settings = Settings.from_env(project_root() / ".env")
    result = register_announcement_factor(
        ResearchStore(settings.database_url),
        default_factors_dir(settings.data_root),
        factor_name=factor_name,
        actor=actor,
    )
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
