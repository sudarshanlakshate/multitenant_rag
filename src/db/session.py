from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


# ============================================================
# SETTINGS
# ============================================================

class Settings(BaseSettings):
    """
    Application configuration.

    Values are loaded from environment variables and .env.
    """

    DATABASE_URL: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()


# ============================================================
# DATABASE ENGINE
# ============================================================

engine = create_engine(
    settings.DATABASE_URL,

    # Production-oriented connection settings.
    pool_pre_ping=True,
    pool_recycle=1800,

    # Don't echo SQL in production.
    echo=False,
)


# ============================================================
# SESSION FACTORY
# ============================================================

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


# ============================================================
# DATABASE SESSION DEPENDENCY
# ============================================================

def get_db():
    """
    Provides a database session.

    The session is automatically closed after use.
    """

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()