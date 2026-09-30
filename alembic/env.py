from __future__ import annotations

import sys
from pathlib import Path
from logging.config import fileConfig

from alembic import context
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine, pool


# ============================================================
# MAKE PROJECT ROOT IMPORTABLE
# ============================================================

# alembic/env.py lives in <project_root>/alembic/, so parents[1] resolves
# to the project root. This allows:
#     from src.db.models import Base
# to work when running alembic from the project root.

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT SQLALCHEMY MODELS
# ============================================================

from src.db.models import Base


# ============================================================
# SETTINGS
# ============================================================

class Settings(BaseSettings):
    """
    Loads configuration from .env.
    """

    DATABASE_URL: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()


# ============================================================
# ALEMBIC CONFIGURATION
# ============================================================

config = context.config


# ------------------------------------------------------------
# Configure Python logging using alembic.ini
# ------------------------------------------------------------

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


# ============================================================
# TARGET METADATA
# ============================================================

# This is what allows Alembic to inspect our SQLAlchemy models.
#
# Alembic compares:
#
# SQLAlchemy models
#       VS
# PostgreSQL database
#
# and generates migration operations.

target_metadata = Base.metadata


# ============================================================
# OFFLINE MIGRATIONS
# ============================================================

def run_migrations_offline() -> None:
    """
    Generate SQL without connecting directly to PostgreSQL.
    """

    context.configure(
        url=settings.DATABASE_URL,
        target_metadata=target_metadata,

        # Render values directly into generated SQL.
        literal_binds=True,

        # PostgreSQL-compatible parameter style.
        dialect_opts={
            "paramstyle": "named",
        },
    )

    with context.begin_transaction():
        context.run_migrations()


# ============================================================
# ONLINE MIGRATIONS
# ============================================================

def run_migrations_online() -> None:
    """
    Connect to PostgreSQL and execute migrations.
    """

    # --------------------------------------------------------
    # Create SQLAlchemy engine
    # --------------------------------------------------------

    connectable = create_engine(
        settings.DATABASE_URL,

        # Alembic migrations don't need a persistent
        # SQLAlchemy connection pool.
        poolclass=pool.NullPool,
    )

    # --------------------------------------------------------
    # Connect to PostgreSQL
    # --------------------------------------------------------

    with connectable.connect() as connection:

        # ----------------------------------------------------
        # Give Alembic the active database connection
        # ----------------------------------------------------

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        # ----------------------------------------------------
        # Execute migration inside transaction
        # ----------------------------------------------------

        with context.begin_transaction():
            context.run_migrations()


# ============================================================
# ENTRY POINT
# ============================================================

if context.is_offline_mode():

    run_migrations_offline()

else:

    run_migrations_online()