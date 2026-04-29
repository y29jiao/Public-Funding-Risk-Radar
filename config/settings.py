"""Project configuration.

Loads environment variables from .env and exposes fixed table names discovered
in step 1. The pipeline now runs in fixed-table mode for demo stability.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")


class Settings:
    """Lightweight settings container."""

    def __init__(self) -> None:
        self.database_url: Optional[str] = os.getenv("DATABASE_URL")
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
        self.search_provider: str = os.getenv("SEARCH_PROVIDER", "mock").strip().lower() or "mock"
        self.search_api_key: Optional[str] = os.getenv("SEARCH_API_KEY") or None
        self.adverse_media_classifier: str = (
            os.getenv("ADVERSE_MEDIA_CLASSIFIER", "mock").strip().lower() or "mock"
        )
        self.entity_disambiguation_classifier: str = (
            os.getenv("ENTITY_DISAMBIGUATION_CLASSIFIER", "mock").strip().lower() or "mock"
        )
        self.review_explanation_provider: str = (
            os.getenv("REVIEW_EXPLANATION_PROVIDER", "mock").strip().lower() or "mock"
        )
        self.investigation_plan_provider: str = (
            os.getenv("INVESTIGATION_PLAN_PROVIDER", "mock").strip().lower() or "mock"
        )
        self.analyst_chat_provider: str = (
            os.getenv("ANALYST_CHAT_PROVIDER", "mock").strip().lower() or "mock"
        )
        try:
            self.search_max_results_per_query: int = int(
                os.getenv("SEARCH_MAX_RESULTS_PER_QUERY", "5")
            )
        except ValueError:
            self.search_max_results_per_query = 5
        try:
            self.adverse_media_classification_limit: int = int(
                os.getenv("ADVERSE_MEDIA_CLASSIFICATION_LIMIT", "500")
            )
        except ValueError:
            self.adverse_media_classification_limit = 500
        try:
            self.entity_disambiguation_limit: int = int(
                os.getenv("ENTITY_DISAMBIGUATION_LIMIT", "500")
            )
        except ValueError:
            self.entity_disambiguation_limit = 500
        try:
            self.review_explanation_limit: int = int(
                os.getenv("REVIEW_EXPLANATION_LIMIT", "100")
            )
        except ValueError:
            self.review_explanation_limit = 100
        try:
            self.investigation_plan_limit: int = int(
                os.getenv("INVESTIGATION_PLAN_LIMIT", "100")
            )
        except ValueError:
            self.investigation_plan_limit = 100

        self.current_year: int = 2026

    def require_database_url(self) -> str:
        if not self.database_url:
            raise RuntimeError(
                "DATABASE_URL is not set. Copy .env.example to .env and fill in "
                "your Postgres connection string."
            )
        return self.database_url


settings = Settings()


FIXED_TABLES: dict[str, str] = {
    "general_golden_records": "general.entity_golden_records",
    "general_entity_funding": "general.vw_entity_funding",
    "general_entity_search": "general.vw_entity_search",
    "fed_grants_contributions": "fed.grants_contributions",
    "fed_grants_decoded": "fed.vw_grants_decoded",
    "ab_grants": "ab.ab_grants",
    "ab_contracts": "ab.ab_contracts",
    "ab_sole_source": "ab.ab_sole_source",
    "ab_non_profit_decoded": "ab.vw_non_profit_decoded",
    "cra_charity_profiles": "cra.vw_charity_profiles",
    "cra_charity_financials_by_year": "cra.vw_charity_financials_by_year",
    "cra_govt_funding_by_charity": "cra.govt_funding_by_charity",
    "cra_t3010_plausibility_flags": "cra.t3010_plausibility_flags",
    "cra_t3010_completeness_issues": "cra.t3010_completeness_issues",
    "cra_t3010_impossibilities": "cra.t3010_impossibilities",
}


# Backward-compatible alias for older helper code. New pipeline code should use
# FIXED_TABLES directly.
MANUAL_TABLES: dict = FIXED_TABLES


# Keywords used by the schema inspector for table/column discovery.
SCHEMA_KEYWORDS: list[str] = [
    "entity", "golden", "match", "cra", "t3010", "charity", "filing",
    "grant", "contribution", "contract", "vendor", "supplier", "recipient",
    "organization", "nonprofit", "sole", "source", "department", "amount",
    "date", "status", "dissolved", "revoked", "risk", "circular",
    "business", "bn", "registration",
]
