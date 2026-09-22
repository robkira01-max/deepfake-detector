"""Alembic env.py — DeepfakeDetector Canada.

Configure la connexion DB depuis DATABASE_URL (env var) et charge
automatiquement les métadonnées de tous les models SQLAlchemy.
"""
from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool

from alembic import context

# Ajouter le backend au path pour que les imports de models fonctionnent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Importer la Base et tous les models pour l'autogenerate
from database import Base  # noqa: E402
import models.user          # noqa: E402, F401
import models.case          # noqa: E402, F401
import models.media_file    # noqa: E402, F401
import models.analysis      # noqa: E402, F401
import models.audit_log     # noqa: E402, F401
import models.report        # noqa: E402, F401

# Config Alembic
config = context.config

# Logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Métadonnées pour l'autogenerate
target_metadata = Base.metadata

# URL de connexion depuis l'environnement (priorité) ou alembic.ini
db_url = os.environ.get("DATABASE_URL") or config.get_main_option("sqlalchemy.url")
if db_url:
    config.set_main_option("sqlalchemy.url", db_url)


def run_migrations_offline() -> None:
    """Mode offline — génère le SQL sans connexion active."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Mode online — connexion active à PostgreSQL."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
