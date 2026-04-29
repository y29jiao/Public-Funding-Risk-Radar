"""Schema discovery utilities.

Pulls table/column metadata from ``information_schema`` and provides helpers
for keyword-based candidate discovery and safe row sampling.
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional

import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# Schemas that we never want to inspect.
EXCLUDED_SCHEMAS = ("pg_catalog", "information_schema", "pg_toast")


def list_tables(engine: Engine) -> pd.DataFrame:
    """Return all user-visible tables and views."""
    excluded = list(EXCLUDED_SCHEMAS)
    stmt = text(
        """
        SELECT table_schema, table_name, table_type
        FROM information_schema.tables
        WHERE table_schema NOT IN :excluded
        ORDER BY table_schema, table_name
        """
    ).bindparams(bindparam("excluded", expanding=True))
    with engine.connect() as conn:
        rows = conn.execute(stmt, {"excluded": excluded}).fetchall()
    df = pd.DataFrame(rows, columns=["table_schema", "table_name", "table_type"])
    logger.info("Discovered %d tables", len(df))
    return df


def list_columns(engine: Engine) -> pd.DataFrame:
    """Return all columns for non-system tables."""
    excluded = list(EXCLUDED_SCHEMAS)
    stmt = text(
        """
        SELECT
            table_schema,
            table_name,
            column_name,
            data_type,
            is_nullable
        FROM information_schema.columns
        WHERE table_schema NOT IN :excluded
        ORDER BY table_schema, table_name, ordinal_position
        """
    ).bindparams(bindparam("excluded", expanding=True))
    with engine.connect() as conn:
        rows = conn.execute(stmt, {"excluded": excluded}).fetchall()
    df = pd.DataFrame(
        rows,
        columns=["table_schema", "table_name", "column_name", "data_type", "is_nullable"],
    )
    logger.info("Discovered %d columns", len(df))
    return df


def find_keyword_matches(
    columns_df: pd.DataFrame,
    keywords: Iterable[str],
) -> pd.DataFrame:
    """Filter columns whose table or column name contains any keyword.

    Returns a copy of ``columns_df`` with an extra ``matched_keywords`` column.
    """
    keywords_lc = [k.lower() for k in keywords]

    def match_keywords(table_name: str, column_name: str) -> list[str]:
        haystack = f"{table_name} {column_name}".lower()
        return [k for k in keywords_lc if k in haystack]

    matched = columns_df.copy()
    matched["matched_keywords"] = [
        match_keywords(t, c)
        for t, c in zip(matched["table_name"], matched["column_name"])
    ]
    return matched[matched["matched_keywords"].map(len) > 0].copy()


def sample_table(
    engine: Engine,
    schema: str,
    table: str,
    limit: int = 5,
) -> Optional[pd.DataFrame]:
    """Sample at most ``limit`` rows from ``schema.table``.

    Never runs SELECT * without LIMIT. Returns ``None`` if the sample fails.
    """
    fq = f'"{schema}"."{table}"'
    sql = text(f"SELECT * FROM {fq} LIMIT :lim").bindparams(lim=int(limit))
    try:
        with engine.connect() as conn:
            df = pd.read_sql(sql, conn)
        return df
    except Exception as exc:  # pragma: no cover - depends on live DB
        logger.warning("Failed to sample %s: %s", fq, exc)
        return None


def categorize_tables(
    matched_columns: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Group candidate tables into the buckets required by step 1."""
    if matched_columns.empty:
        return {}

    # Build a per-table summary of matched keywords.
    tbl = (
        matched_columns
        .assign(full_table=lambda d: d["table_schema"] + "." + d["table_name"])
        .groupby(["table_schema", "table_name", "full_table"])["matched_keywords"]
        .apply(lambda lists: sorted({kw for sub in lists for kw in sub}))
        .reset_index()
        .rename(columns={"matched_keywords": "keywords"})
    )

    def has_any(kws: list[str], terms: Iterable[str]) -> bool:
        return any(t in kws for t in terms)

    buckets: dict[str, pd.DataFrame] = {}

    buckets["Candidate Entity Tables"] = tbl[
        tbl["keywords"].apply(
            lambda k: has_any(k, ("entity", "organization", "nonprofit", "recipient", "vendor", "supplier"))
        )
    ].copy()

    buckets["Candidate General / Golden Match Tables"] = tbl[
        tbl["keywords"].apply(lambda k: has_any(k, ("golden", "match")))
        | tbl["table_schema"].str.lower().eq("general")
    ].copy()

    buckets["Candidate CRA Tables"] = tbl[
        tbl["keywords"].apply(lambda k: has_any(k, ("cra", "t3010", "charity")))
        | tbl["table_schema"].str.lower().eq("cra")
    ].copy()

    buckets["Candidate FED Funding Tables"] = tbl[
        tbl["keywords"].apply(lambda k: has_any(k, ("grant", "contribution")))
        | tbl["table_schema"].str.lower().isin(["fed", "federal"])
    ].copy()

    buckets["Candidate AB Contract / Grant Tables"] = tbl[
        tbl["keywords"].apply(lambda k: has_any(k, ("contract", "sole", "vendor", "supplier")))
        | tbl["table_schema"].str.lower().isin(["ab", "alberta"])
    ].copy()

    buckets["Candidate Status / Filing Tables"] = tbl[
        tbl["keywords"].apply(
            lambda k: has_any(k, ("status", "filing", "dissolved", "revoked"))
        )
    ].copy()

    # Special case: amount/date column-level (not table-level)
    amount_date_cols = matched_columns[
        matched_columns["matched_keywords"].apply(
            lambda kws: any(t in kws for t in ("amount", "date"))
        )
    ][["table_schema", "table_name", "column_name", "data_type", "matched_keywords"]].copy()
    buckets["Candidate Amount / Date Columns"] = amount_date_cols

    return buckets
