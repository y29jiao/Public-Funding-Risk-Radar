"""Vendor concentration metrics focused on Alberta contracts and sole-source."""
from __future__ import annotations

import logging

import pandas as pd

from db.connection import get_engine
from db.queries import run_aggregation
from src.entity_resolution import normalize_entity_name
from src.utils import ensure_output_columns, stable_hash

logger = logging.getLogger(__name__)

MARKET_SHARE_COLUMNS = [
    "market_id", "market_definition", "source", "department", "category", "year",
    "entity_id", "vendor_name", "vendor_spending", "vendor_contract_count",
    "total_market_spending", "vendor_share", "vendor_rank",
]

CONCENTRATION_COLUMNS = [
    "market_id", "market_definition", "source", "department", "category", "year",
    "total_spending", "contract_count", "vendor_count", "top_1_vendor",
    "top_1_share", "top_5_share", "hhi", "concentration_score",
    "concentration_risk_level",
]


def load_vendor_records() -> pd.DataFrame:
    """Load v1 vendor records from AB contracts and AB sole-source only."""
    engine = get_engine()
    sql = """
        SELECT
            'AB_CONTRACTS' AS source,
            ministry AS department,
            'contract' AS category,
            CAST(NULLIF(SUBSTRING(display_fiscal_year FROM 1 FOR 4), '') AS int) AS year,
            recipient AS vendor_name,
            CAST(NULL AS text) AS entity_id,
            amount AS amount,
            id::text AS contract_id
        FROM ab.ab_contracts
        WHERE recipient IS NOT NULL AND ministry IS NOT NULL AND amount IS NOT NULL
        UNION ALL
        SELECT
            'AB_SOLE_SOURCE' AS source,
            ministry AS department,
            COALESCE(NULLIF(permitted_situations, ''), 'sole_source') AS category,
            CAST(NULLIF(SUBSTRING(display_fiscal_year FROM 1 FOR 4), '') AS int) AS year,
            vendor AS vendor_name,
            CAST(NULL AS text) AS entity_id,
            amount AS amount,
            COALESCE(contract_number, id::text) AS contract_id
        FROM ab.ab_sole_source
        WHERE vendor IS NOT NULL AND ministry IS NOT NULL AND amount IS NOT NULL
    """
    return run_aggregation(engine, sql)


def compute_vendor_concentration(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if records is None or records.empty:
        logger.warning("No vendor records available; writing empty concentration outputs.")
        return (
            ensure_output_columns(pd.DataFrame(), MARKET_SHARE_COLUMNS),
            ensure_output_columns(pd.DataFrame(), CONCENTRATION_COLUMNS),
        )

    df = records.copy()
    for col in ["source", "department", "category", "year", "vendor_name", "entity_id", "amount", "contract_id"]:
        if col not in df.columns:
            df[col] = pd.NA

    missing_vendor = int(df["vendor_name"].isna().sum())
    missing_amount = int(df["amount"].isna().sum())
    missing_category = int(df["category"].isna().sum())
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    before = len(df)
    df = df[df["amount"].gt(0) & df["vendor_name"].notna() & df["department"].notna() & df["year"].notna()].copy()
    logger.warning("Vendor concentration dropped %d rows due to missing vendor.", missing_vendor)
    logger.warning("Vendor concentration dropped %d rows due to missing/nonpositive amount.", before - len(df))
    logger.warning("Vendor concentration found %d rows with missing category.", missing_category)

    df["category"] = df["category"].fillna("").astype(str).str.strip()
    fallback = df["category"].eq("")
    df["market_definition"] = "source+department+category+year"
    df.loc[fallback, "market_definition"] = "source+department+year"
    logger.warning("%d vendor rows use fallback market definition.", int(fallback.sum()))
    df["market_id"] = [
        stable_hash(row.source, row.department, row.category, row.year)
        if row.category else stable_hash(row.source, row.department, row.year)
        for row in df.itertuples(index=False)
    ]
    df["entity_id"] = df["entity_id"].where(
        df["entity_id"].notna() & df["entity_id"].astype(str).str.len().gt(0),
        df["vendor_name"].astype(str).map(lambda n: "vendor:" + stable_hash(normalize_entity_name(n))),
    )

    vendor = (
        df.groupby(["market_id", "market_definition", "source", "department", "category", "year", "entity_id", "vendor_name"], dropna=False)
        .agg(vendor_spending=("amount", "sum"), vendor_contract_count=("contract_id", "nunique"))
        .reset_index()
    )
    totals = vendor.groupby("market_id", dropna=False).agg(total_market_spending=("vendor_spending", "sum")).reset_index()
    market_shares = vendor.merge(totals, on="market_id", how="left")
    market_shares["vendor_share"] = market_shares["vendor_spending"] / market_shares["total_market_spending"]
    market_shares["vendor_rank"] = market_shares.groupby("market_id")["vendor_share"].rank(method="first", ascending=False).astype(int)

    concentration = (
        market_shares.groupby(["market_id", "market_definition", "source", "department", "category", "year"], dropna=False)
        .apply(_market_metrics)
        .reset_index()
    )
    return (
        ensure_output_columns(market_shares, MARKET_SHARE_COLUMNS),
        ensure_output_columns(concentration, CONCENTRATION_COLUMNS),
    )


def _market_metrics(group: pd.DataFrame) -> pd.Series:
    ordered = group.sort_values("vendor_share", ascending=False)
    top1 = float(ordered["vendor_share"].iloc[0])
    top5 = float(ordered["vendor_share"].head(5).sum())
    hhi = float((ordered["vendor_share"] ** 2).sum())
    top1_score = min(top1 / 0.50, 1.0) * 100
    top5_score = min(top5 / 0.90, 1.0) * 100
    hhi_score = min(hhi / 0.25, 1.0) * 100
    score = 0.40 * top1_score + 0.30 * top5_score + 0.30 * hhi_score
    if hhi >= 0.25 or top1 >= 0.50 or top5 >= 0.90:
        risk = "Very High"
    elif hhi >= 0.18 or top1 >= 0.35 or top5 >= 0.75:
        risk = "High"
    elif hhi >= 0.10 or top1 >= 0.20 or top5 >= 0.60:
        risk = "Medium"
    else:
        risk = "Low"
    return pd.Series({
        "total_spending": ordered["vendor_spending"].sum(),
        "contract_count": ordered["vendor_contract_count"].sum(),
        "vendor_count": ordered["vendor_name"].nunique(),
        "top_1_vendor": ordered["vendor_name"].iloc[0],
        "top_1_share": top1,
        "top_5_share": top5,
        "hhi": hhi,
        "concentration_score": score,
        "concentration_risk_level": risk,
    })
