from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from alembic import command
from alembic.config import Config

from quant_data.config import Settings

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


if __name__ == "__main__":
    app()
