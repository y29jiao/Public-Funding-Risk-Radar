"""Shared helpers: logging, CSV I/O, simple type coercion."""
from __future__ import annotations

import csv
import hashlib
import logging
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

from config.settings import OUTPUTS_DIR


def configure_logging(level: int = logging.INFO) -> None:
    """Idempotent log configuration with a clear, single-line format."""
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(level)


def output_path(filename: str) -> Path:
    return OUTPUTS_DIR / filename


def write_csv(df: pd.DataFrame, filename: str) -> Path:
    """Write a DataFrame as UTF-8 CSV under ``outputs/``."""
    path = output_path(filename)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
    return path


def read_csv_safe(filename: str) -> Optional[pd.DataFrame]:
    """Read a CSV from ``outputs/``; return ``None`` if missing or empty."""
    path = output_path(filename)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return df
    except pd.errors.EmptyDataError:
        return None


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Ensure each column exists in ``df``; add missing ones with NaN."""
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def ensure_output_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Ensure columns exist and return them in the requested order."""
    df = ensure_columns(df, columns)
    return df[list(columns)].copy()


def stable_hash(*parts: Any) -> str:
    """Short deterministic hash usable as a synthetic ID."""
    h = hashlib.sha1()
    for p in parts:
        h.update(("|" + str(p)).encode("utf-8"))
    return h.hexdigest()[:16]


def coerce_year(value: Any) -> Optional[int]:
    """Best-effort year extraction from int/str/datetime values."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int,)) and 1900 < value < 2100:
        return int(value)
    try:
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            return None
        return int(ts.year)
    except Exception:
        return None


def truthy(value: Any) -> bool:
    """Normalize common database truthy values."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "sole", "sole source"}


def normalize_colname(value: str) -> str:
    """Lowercase a column name and remove non-alphanumerics."""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    """Find a likely column by exact or fuzzy normalized-name matching."""
    cols = list(columns)
    norm_to_col = {normalize_colname(c): c for c in cols}
    for candidate in candidates:
        norm = normalize_colname(candidate)
        if norm in norm_to_col:
            return norm_to_col[norm]
    for candidate in candidates:
        norm = normalize_colname(candidate)
        for col_norm, col in norm_to_col.items():
            if norm and (norm in col_norm or col_norm in norm):
                return col
    return None
