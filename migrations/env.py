from __future__ import annotations

import logging
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool, text

from quant_data.database import metadata

config = context.config
if config.config_file_name is not None:
    # Alembic is also invoked in-process by the API/tests.  ``fileConfig``'s
    # defaults disable every already-created application logger and replace
    # the host process' root handlers; that made later warnings disappear and
    # broke any embedding application's logging/capture pipeline.  Preserve an
    # existing host configuration, while still installing alembic.ini logging
    # for the standalone CLI where no handlers exist yet.
    root_logger = logging.getLogger()
    host_handlers = list(root_logger.handlers)
    host_level = root_logger.level
    fileConfig(config.config_file_name, disable_existing_loggers=False)
    if host_handlers:
        root_logger.handlers[:] = host_handlers
        root_logger.setLevel(host_level)

# Programmatic callers (tests, restore validation and release preflight) pass an
# explicit database URL through the Alembic Config.  It must win over the host
# process environment; otherwise an isolated migration can silently target the
# production database named by DATABASE_URL.
database_url = config.get_main_option("sqlalchemy.url") or os.environ.get(
    "DATABASE_URL"
)
if not database_url:
    raise RuntimeError("DATABASE_URL is required for database migrations")
config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
target_metadata = metadata


def run_migrations_offline() -> None:
    context.configure(
        url=database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_schemas=True,
        version_table_schema="quantlab",
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        connection.execute(text("CREATE SCHEMA IF NOT EXISTS quantlab"))
        connection.commit()
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_schemas=True,
            version_table_schema="quantlab",
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
