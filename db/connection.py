"""SQLAlchemy engine factory."""
from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from config.settings import settings

logger = logging.getLogger(__name__)

_engine: Optional[Engine] = None


def get_engine() -> Engine:
    """Return a cached SQLAlchemy engine for the configured DATABASE_URL."""
    global _engine
    if _engine is not None:
        return _engine

    url = settings.require_database_url()
    logger.info("Creating SQLAlchemy engine for %s", _redact(url))
    _engine = create_engine(url, pool_pre_ping=True, future=True)
    return _engine


def _redact(url: str) -> str:
    """Hide credentials in a connection URL for logging."""
    try:
        if "@" not in url:
            return url
        scheme_userpass, host_part = url.split("@", 1)
        if "://" not in scheme_userpass:
            return url
        scheme, _ = scheme_userpass.split("://", 1)
        return f"{scheme}://***:***@{host_part}"
    except Exception:
        return "<redacted>"
