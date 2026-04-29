"""Query helpers used by later pipeline steps.

These helpers push aggregation into SQL where possible so we never pull
full tables into Python memory.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


def fq(schema: Optional[str], table: str) -> str:
    """Quote a table identifier as ``"schema"."table"`` or ``"table"``."""
    if schema:
        return f'"{schema}"."{table}"'
    return f'"{table}"'


def run_aggregation(
    engine: Engine,
    sql: str,
    params: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    """Run a SQL aggregation and return a DataFrame.

    Logs (and returns an empty DataFrame on) failures so callers can degrade
    gracefully rather than crashing the pipeline.
    """
    params = params or {}
    try:
        with engine.connect() as conn:
            stmt = text(sql).bindparams(**params) if params else text(sql)
            df = pd.read_sql(stmt, conn)
        return df
    except Exception as exc:  # pragma: no cover - depends on live DB
        logger.warning("Aggregation failed: %s\nSQL: %s", exc, sql)
        return pd.DataFrame()


def get_table_row_count(
    engine: Engine,
    schema: str,
    table: str,
) -> Optional[int]:
    """Approximate row count using ``pg_class.reltuples`` (cheap, fast)."""
    sql = text(
        """
        SELECT reltuples::bigint AS approx_rows
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = :schema AND c.relname = :table
        """
    ).bindparams(schema=schema, table=table)
    try:
        with engine.connect() as conn:
            row = conn.execute(sql).fetchone()
        if row is None:
            return None
        return int(row[0])
    except Exception as exc:  # pragma: no cover
        logger.warning("Row count failed for %s.%s: %s", schema, table, exc)
        return None
