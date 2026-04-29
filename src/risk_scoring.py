"""Combine screening signals into demo-ready entity risk scores."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.utils import ensure_columns, ensure_output_columns

logger = logging.getLogger(__name__)

RISK_SCORE_COLUMNS = [
    "entity_id",
    "canonical_name",
    "normalized_name",
    "source_systems",
    "province",
    "city",
    "total_public_funding",
    "gross_positive_funding",
    "net_funding",
    "zombie_score",
    "max_vendor_share",
    "max_concentration_score",
    "overall_risk_score",
    "risk_level",
    "review_priority",
    "explanation",
]


def build_risk_scores(
    entity: pd.DataFrame,
    zombie: pd.DataFrame,
    market_shares: pd.DataFrame,
    concentration: pd.DataFrame,
) -> pd.DataFrame:
    """Build one row per entity with a composite screening score."""
    entity = ensure_columns(entity if entity is not None else pd.DataFrame(), RISK_SCORE_COLUMNS)
    if entity.empty:
        logger.warning("No entity rows available for risk scoring.")
        return ensure_output_columns(pd.DataFrame(), RISK_SCORE_COLUMNS)

    zombie = ensure_columns(zombie if zombie is not None else pd.DataFrame(), ["entity_id", "zombie_score", "zombie_flags"])
    market_shares = ensure_columns(market_shares if market_shares is not None else pd.DataFrame(), ["entity_id", "vendor_share", "market_id"])
    concentration = ensure_columns(concentration if concentration is not None else pd.DataFrame(), ["market_id", "concentration_score", "concentration_risk_level"])
    zombie["entity_id"] = zombie["entity_id"].astype(str)
    market_shares["entity_id"] = market_shares["entity_id"].astype(str)
    concentration["market_id"] = concentration["market_id"].astype(str)
    market_shares["market_id"] = market_shares["market_id"].astype(str)

    df = entity.copy()
    df["entity_id"] = df["entity_id"].astype(str)
    df["total_public_funding"] = pd.to_numeric(df["total_public_funding"], errors="coerce").fillna(0)
    if "gross_positive_funding" not in df.columns:
        df["gross_positive_funding"] = df["total_public_funding"]
    if "net_funding" not in df.columns:
        df["net_funding"] = df["total_public_funding"]

    zombie_small = zombie[["entity_id", "zombie_score", "zombie_flags"]].drop_duplicates("entity_id")
    df = df.merge(zombie_small, on="entity_id", how="left", suffixes=("", "_z"))
    if "zombie_score_z" in df.columns:
        df["zombie_score"] = df["zombie_score"].fillna(df["zombie_score_z"])
    df["zombie_score"] = pd.to_numeric(df["zombie_score"], errors="coerce").fillna(0)

    vendor = market_shares.merge(concentration[["market_id", "concentration_score"]], on="market_id", how="left")
    if vendor.empty:
        vendor_metrics = pd.DataFrame(columns=["entity_id", "canonical_name", "max_vendor_share", "max_concentration_score"])
    else:
        vendor["vendor_share"] = pd.to_numeric(vendor["vendor_share"], errors="coerce")
        vendor["concentration_score"] = pd.to_numeric(vendor["concentration_score"], errors="coerce")
        vendor_metrics = (
            vendor.groupby("entity_id", dropna=False)
            .agg(
                canonical_name=("vendor_name", "first"),
                max_vendor_share=("vendor_share", "max"),
                max_concentration_score=("concentration_score", "max"),
            )
            .reset_index()
        )
    df = df.merge(vendor_metrics, on="entity_id", how="left", suffixes=("", "_vendor"))
    for col in ["max_vendor_share", "max_concentration_score"]:
        other = f"{col}_vendor"
        if other in df.columns:
            df[col] = df[col].fillna(df[other])
            df = df.drop(columns=[other])
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    missing_vendor_entities = vendor_metrics[
        ~vendor_metrics["entity_id"].isin(df["entity_id"])
    ].copy()
    if not missing_vendor_entities.empty:
        vendor_only = pd.DataFrame({
            "entity_id": missing_vendor_entities["entity_id"],
            "canonical_name": missing_vendor_entities["canonical_name"],
            "normalized_name": missing_vendor_entities["canonical_name"].astype(str).str.lower(),
            "source_systems": "AB_VENDOR",
            "province": pd.NA,
            "city": pd.NA,
            "total_public_funding": 0,
            "gross_positive_funding": 0,
            "net_funding": 0,
            "zombie_score": 0,
            "max_vendor_share": missing_vendor_entities["max_vendor_share"],
            "max_concentration_score": missing_vendor_entities["max_concentration_score"],
        })
        df = pd.concat([df, vendor_only], ignore_index=True)

    funding_score = _funding_score(df["total_public_funding"])
    concentration_score = df["max_concentration_score"].clip(lower=0, upper=100)
    df["overall_risk_score"] = (
        0.45 * df["zombie_score"].clip(lower=0, upper=100)
        + 0.35 * concentration_score
        + 0.20 * funding_score
    ).round(1)
    df["risk_level"] = pd.cut(
        df["overall_risk_score"],
        bins=[-1, 24, 49, 74, 100],
        labels=["Low", "Medium", "High", "Very High"],
    ).astype(str)
    df["review_priority"] = pd.cut(
        df["overall_risk_score"],
        bins=[-1, 39, 69, 100],
        labels=["Routine", "Prioritized", "Immediate"],
    ).astype(str)
    df["explanation"] = df.apply(_explain_row, axis=1)
    return ensure_output_columns(df, RISK_SCORE_COLUMNS)


def _funding_score(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0)
    return (np.log10(values.clip(lower=0) + 1) / 7 * 100).clip(lower=0, upper=100)


def _explain_row(row: pd.Series) -> str:
    reasons = []
    if row.get("zombie_score", 0) >= 50:
        reasons.append("zombie-recipient indicators")
    if row.get("max_concentration_score", 0) >= 50:
        reasons.append("vendor concentration indicators")
    if row.get("max_vendor_share", 0) >= 0.35:
        reasons.append("large vendor market share")
    if row.get("total_public_funding", 0) >= 1_000_000:
        reasons.append("high public funding exposure")
    if not reasons:
        reasons.append("limited currently observed risk indicators")
    return "Flagged for further review due to " + ", ".join(reasons) + "."
