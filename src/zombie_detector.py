"""Zombie-recipient screening signals.

Scores are screening flags only and do not assert wrongdoing.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from config.settings import settings
from src.utils import coerce_year, ensure_output_columns

logger = logging.getLogger(__name__)

ZOMBIE_COLUMNS = [
    "entity_id",
    "canonical_name",
    "total_public_funding",
    "first_funding_date",
    "last_funding_date",
    "registry_status",
    "cra_latest_status",
    "ab_non_profit_status",
    "normalized_status",
    "cra_latest_filing_year",
    "years_since_latest_filing",
    "stopped_filing_after_funding",
    "public_funding_dependency_ratio",
    "zombie_score",
    "zombie_risk_level",
    "zombie_flags",
]


def normalize_status(*statuses: Any) -> str:
    text = " ".join(str(s).lower() for s in statuses if s is not None and not pd.isna(s))
    if not text.strip():
        return "unknown"
    if "revok" in text:
        return "revoked"
    if "dissol" in text or "struck" in text:
        return "dissolved"
    if "suspend" in text:
        return "suspended"
    if "inactive" in text or "closed" in text or "cancel" in text or "cancelled" in text:
        return "inactive"
    if "active" in text or "registered" in text or "good standing" in text:
        return "active"
    return "unknown"


def compute_zombie_signals(entity_df: pd.DataFrame) -> pd.DataFrame:
    if entity_df is None or entity_df.empty:
        logger.warning("No entity funding summary rows available for zombie signals.")
        return ensure_output_columns(pd.DataFrame(), ZOMBIE_COLUMNS)

    df = entity_df.copy()
    for col in [
        "registry_status", "cra_latest_status", "ab_non_profit_status", "cra_latest_filing_year",
        "last_funding_date", "public_funding_dependency_ratio",
        "cra_government_revenue_latest", "cra_total_revenue_latest",
        "total_public_funding", "is_charity",
    ]:
        if col not in df.columns:
            df[col] = pd.NA

    df["normalized_status"] = [
        normalize_status(reg, cra, ab)
        for reg, cra, ab in zip(df["registry_status"], df["cra_latest_status"], df["ab_non_profit_status"])
    ]
    df["cra_latest_filing_year"] = pd.to_numeric(df["cra_latest_filing_year"], errors="coerce")
    df["last_funding_year"] = df["last_funding_date"].map(coerce_year)
    df["years_since_latest_filing"] = settings.current_year - df["cra_latest_filing_year"]
    df.loc[df["cra_latest_filing_year"].isna(), "years_since_latest_filing"] = pd.NA
    df["is_charity_bool"] = df["is_charity"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
    df["stopped_filing_after_funding"] = (
        df["is_charity_bool"]
        & df["cra_latest_filing_year"].notna()
        & df["last_funding_year"].notna()
        & (df["cra_latest_filing_year"] < df["last_funding_year"])
    )

    dep = pd.to_numeric(df["public_funding_dependency_ratio"], errors="coerce")
    missing_dep = dep.isna()
    dep_calc = (
        pd.to_numeric(df["cra_government_revenue_latest"], errors="coerce")
        / pd.to_numeric(df["cra_total_revenue_latest"], errors="coerce")
    )
    df["public_funding_dependency_ratio"] = dep.where(~missing_dep, dep_calc)
    df["public_funding_dependency_ratio"] = df["public_funding_dependency_ratio"].replace([float("inf")], pd.NA)

    scores = []
    flags_list = []
    for _, row in df.iterrows():
        score = 0
        flags: list[str] = []
        status = row.get("normalized_status", "unknown")
        if status in {"inactive", "revoked", "dissolved", "suspended"}:
            score += 25
            flags.append(f"status_{status}")
        if bool(row.get("stopped_filing_after_funding")):
            score += 25
            flags.append("stopped_filing_after_funding")
        years = row.get("years_since_latest_filing")
        if pd.notna(years) and years >= 2:
            score += 15
            flags.append("latest_filing_2plus_years_old")
        dependency = row.get("public_funding_dependency_ratio")
        if pd.notna(dependency):
            if dependency >= 0.80:
                score += 25
                flags.append("dependency_80pct_plus")
            elif dependency >= 0.70:
                score += 18
                flags.append("dependency_70pct_plus")
            elif dependency >= 0.50:
                score += 10
                flags.append("dependency_50pct_plus")
        funding = pd.to_numeric(row.get("total_public_funding"), errors="coerce")
        if pd.notna(funding) and funding >= 1_000_000:
            score += 10
            flags.append("funding_1m_plus")
        scores.append(min(int(score), 100))
        flags_list.append(";".join(flags))

    df["zombie_score"] = scores
    df["zombie_flags"] = flags_list
    df["zombie_risk_level"] = pd.cut(
        df["zombie_score"],
        bins=[-1, 24, 49, 74, 100],
        labels=["Low", "Medium", "High", "Very High"],
    ).astype(str)
    return ensure_output_columns(df, ZOMBIE_COLUMNS)
