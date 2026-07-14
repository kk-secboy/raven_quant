from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from quant_platform.db_cli import upgrade_database

from .app import create_app
from .config import GatewaySettings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Windows QMT sandbox gateway")
    parser.add_argument("--env-file", type=Path, default=Path("deploy/qmt-gateway.env"))
    parser.add_argument("--no-migrate", action="store_true")
    arguments = parser.parse_args()
    settings = GatewaySettings.from_env(arguments.env_file)
    if not arguments.no_migrate:
        upgrade_database(settings.database_url)
    uvicorn.run(
        create_app(settings),
        host=settings.bind_host,
        port=settings.bind_port,
        access_log=False,
    )


if __name__ == "__main__":
    main()
