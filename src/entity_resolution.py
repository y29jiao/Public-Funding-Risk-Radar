"""Entity normalization and synthetic ID helpers."""
from __future__ import annotations

import re
from typing import Any, Optional

import pandas as pd
from rapidfuzz import fuzz

from src.utils import stable_hash

LEGAL_SUFFIXES = {
    "inc",
    "ltd",
    "limited",
    "corp",
    "corporation",
    "incorporated",
    "society",
    "association",
    "co",
    "company",
}


def normalize_entity_name(name: Any) -> str:
    """Normalize an organization name for matching without making legal claims."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    text = str(name).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    parts = [p for p in text.split() if p not in LEGAL_SUFFIXES]
    return " ".join(parts)


def normalize_identifier(value: Any) -> Optional[str]:
    """Normalize business/charity numbers to digits where possible."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = re.sub(r"\D+", "", str(value))
    return text or None


def make_entity_id(
    golden_id: Any = None,
    business_number: Any = None,
    charity_number: Any = None,
    normalized_name: str = "",
    province: Any = None,
) -> str:
    """Create an entity ID using the requested priority order."""
    if golden_id is not None and not pd.isna(golden_id) and str(golden_id).strip():
        return f"golden:{str(golden_id).strip()}"
    bn = normalize_identifier(business_number)
    if bn:
        return f"bn:{bn}"
    charity = normalize_identifier(charity_number)
    if charity:
        return f"charity:{charity}"
    prov = "" if province is None or pd.isna(province) else str(province).strip().upper()
    if normalized_name and prov:
        return f"name_prov:{stable_hash(normalized_name, prov)}"
    return f"name:{stable_hash(normalized_name)}"


def similar_name_score(left: str, right: str) -> float:
    """Return a 0-1 token-sort similarity for optional diagnostics."""
    if not left or not right:
        return 0.0
    return fuzz.token_sort_ratio(left, right) / 100.0
