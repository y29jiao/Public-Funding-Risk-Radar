"""Build the entity-level public funding summary in fixed-table mode."""
from __future__ import annotations

import logging

import pandas as pd

from db.connection import get_engine
from db.queries import run_aggregation
from src.entity_resolution import normalize_entity_name
from src.utils import ensure_output_columns

logger = logging.getLogger(__name__)

ENTITY_FUNDING_COLUMNS = [
    "entity_id",
    "canonical_name",
    "normalized_name",
    "entity_type",
    "source_systems",
    "province",
    "city",
    "total_public_funding",
    "gross_positive_funding",
    "net_funding",
    "fed_total_funding",
    "ab_total_funding",
    "cra_government_revenue",
    "cra_total_revenue_latest",
    "cra_government_revenue_latest",
    "public_funding_dependency_ratio",
    "first_funding_date",
    "last_funding_date",
    "funding_year_count",
    "funding_record_count",
    "top_department",
    "top_program",
    "top_category",
    "is_charity",
    "charity_number",
    "business_number",
    "cra_latest_filing_year",
    "cra_latest_status",
    "registry_status",
    "ab_non_profit_status",
    "ab_non_profit_status_description",
    "vendor_total_contract_value",
    "vendor_contract_count",
    "sole_source_contract_value",
    "sole_source_contract_count",
]


def build_entity_funding_summary() -> pd.DataFrame:
    """Use discovered core views/tables rather than table-name inference."""
    engine = get_engine()
    sql = """
        WITH latest_cra_profile AS (
            SELECT *
            FROM (
                SELECT
                    bn,
                    legal_name,
                    city,
                    province,
                    ROW_NUMBER() OVER (PARTITION BY bn ORDER BY fiscal_year DESC NULLS LAST) AS rn
                FROM cra.vw_charity_profiles
            ) ranked
            WHERE rn = 1
        ),
        latest_cra_govt AS (
            SELECT *
            FROM (
                SELECT
                    bn,
                    fiscal_year,
                    legal_name,
                    total_govt,
                    revenue,
                    govt_share_of_rev,
                    ROW_NUMBER() OVER (PARTITION BY bn ORDER BY fiscal_year DESC NULLS LAST) AS rn
                FROM cra.govt_funding_by_charity
            ) ranked
            WHERE rn = 1
        ),
        cra_govt_all_years AS (
            SELECT
                bn,
                SUM(COALESCE(total_govt, 0)) AS cra_government_revenue
            FROM cra.govt_funding_by_charity
            GROUP BY bn
        ),
        ab_non_profit AS (
            SELECT
                LOWER(REGEXP_REPLACE(legal_name, '[^a-zA-Z0-9]+', ' ', 'g')) AS normalized_legal_name,
                MAX(status) AS ab_non_profit_status,
                MAX(status_description) AS ab_non_profit_status_description,
                MAX(city) AS ab_city
            FROM ab.vw_non_profit_decoded
            GROUP BY 1
        )
        SELECT
            CAST(COALESCE(g.id, f.entity_id) AS text) AS entity_id,
            COALESCE(g.canonical_name, f.canonical_name) AS canonical_name,
            COALESCE(g.norm_name, LOWER(REGEXP_REPLACE(f.canonical_name, '[^a-zA-Z0-9]+', ' ', 'g'))) AS normalized_name,
            COALESCE(g.entity_type, f.entity_type) AS entity_type,
            COALESCE(CAST(g.dataset_sources AS text), CAST(f.dataset_sources AS text)) AS source_systems,
            p.province AS province,
            COALESCE(p.city, np.ab_city) AS city,
            COALESCE(latest_govt.total_govt, 0)
                + GREATEST(COALESCE(f.fed_total_grants, 0), 0)
                + GREATEST(COALESCE(f.ab_total_grants, 0), 0)
                + GREATEST(COALESCE(f.ab_total_contracts, 0), 0)
                + GREATEST(COALESCE(f.ab_total_sole_source, 0), 0) AS total_public_funding,
            COALESCE(latest_govt.total_govt, 0)
                + GREATEST(COALESCE(f.fed_total_grants, 0), 0)
                + GREATEST(COALESCE(f.ab_total_grants, 0), 0)
                + GREATEST(COALESCE(f.ab_total_contracts, 0), 0)
                + GREATEST(COALESCE(f.ab_total_sole_source, 0), 0) AS gross_positive_funding,
            COALESCE(latest_govt.total_govt, 0)
                + COALESCE(f.fed_total_grants, 0)
                + COALESCE(f.ab_total_grants, 0)
                + COALESCE(f.ab_total_contracts, 0)
                + COALESCE(f.ab_total_sole_source, 0) AS net_funding,
            COALESCE(f.fed_total_grants, 0) AS fed_total_funding,
            COALESCE(f.ab_total_grants, 0) + COALESCE(f.ab_total_contracts, 0) + COALESCE(f.ab_total_sole_source, 0) AS ab_total_funding,
            COALESCE(govt_all.cra_government_revenue, 0) AS cra_government_revenue,
            latest_govt.revenue AS cra_total_revenue_latest,
            latest_govt.total_govt AS cra_government_revenue_latest,
            latest_govt.govt_share_of_rev AS public_funding_dependency_ratio,
            LEAST(
                f.fed_earliest_grant,
                MAKE_DATE(f.cra_earliest_year, 1, 1)
            ) AS first_funding_date,
            GREATEST(
                f.fed_latest_grant,
                MAKE_DATE(f.cra_latest_year, 12, 31)
            ) AS last_funding_date,
            COALESCE(f.cra_filing_count, 0)
                + COALESCE(f.fed_grant_count, 0)
                + COALESCE(f.ab_grant_payment_count, 0)
                + COALESCE(f.ab_contract_count, 0)
                + COALESCE(f.ab_sole_source_count, 0) AS funding_year_count,
            COALESCE(f.cra_filing_count, 0)
                + COALESCE(f.fed_grant_count, 0)
                + COALESCE(f.ab_grant_payment_count, 0)
                + COALESCE(f.ab_contract_count, 0)
                + COALESCE(f.ab_sole_source_count, 0) AS funding_record_count,
            NULL::text AS top_department,
            NULL::text AS top_program,
            NULL::text AS top_category,
            CASE
                WHEN COALESCE(g.entity_type, f.entity_type) ILIKE '%charit%' THEN TRUE
                WHEN CAST(COALESCE(g.dataset_sources, f.dataset_sources) AS text) ILIKE '%cra%' THEN TRUE
                ELSE FALSE
            END AS is_charity,
            COALESCE(f.bn_root, latest_govt.bn, p.bn) AS charity_number,
            f.bn_root AS business_number,
            f.cra_latest_year AS cra_latest_filing_year,
            NULL::text AS cra_latest_status,
            COALESCE(g.status, f.status) AS registry_status,
            np.ab_non_profit_status,
            np.ab_non_profit_status_description,
            COALESCE(f.ab_total_contracts, 0) + COALESCE(f.ab_total_sole_source, 0) AS vendor_total_contract_value,
            COALESCE(f.ab_contract_count, 0) + COALESCE(f.ab_sole_source_count, 0) AS vendor_contract_count,
            COALESCE(f.ab_total_sole_source, 0) AS sole_source_contract_value,
            COALESCE(f.ab_sole_source_count, 0) AS sole_source_contract_count
        FROM general.vw_entity_funding f
        LEFT JOIN general.entity_golden_records g
            ON g.id = f.entity_id
        LEFT JOIN latest_cra_profile p
            ON p.bn = f.bn_root
        LEFT JOIN latest_cra_govt latest_govt
            ON latest_govt.bn = f.bn_root
        LEFT JOIN cra_govt_all_years govt_all
            ON govt_all.bn = f.bn_root
        LEFT JOIN ab_non_profit np
            ON np.normalized_legal_name = LOWER(REGEXP_REPLACE(COALESCE(g.canonical_name, f.canonical_name), '[^a-zA-Z0-9]+', ' ', 'g'))
    """
    df = run_aggregation(engine, sql)
    if df.empty:
        logger.warning("Fixed-table entity funding query returned no rows.")
        return ensure_output_columns(pd.DataFrame(), ENTITY_FUNDING_COLUMNS)

    df["normalized_name"] = df["canonical_name"].map(normalize_entity_name)
    for col in [
        "total_public_funding", "gross_positive_funding", "net_funding",
        "fed_total_funding", "ab_total_funding", "cra_government_revenue",
        "cra_total_revenue_latest", "cra_government_revenue_latest",
        "public_funding_dependency_ratio", "vendor_total_contract_value",
        "vendor_contract_count", "sole_source_contract_value",
        "sole_source_contract_count",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return ensure_output_columns(df, ENTITY_FUNDING_COLUMNS)
