"""Adverse-media candidate selection and deterministic mock search."""
from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel

from config.settings import settings
from src.entity_resolution import normalize_entity_name
from src.utils import ensure_columns, ensure_output_columns, stable_hash

logger = logging.getLogger(__name__)

CANDIDATE_COLUMNS = [
    "entity_id", "canonical_name", "normalized_name", "candidate_reason",
    "total_public_funding", "zombie_score", "max_vendor_share",
    "max_concentration_score", "source_systems", "province", "city",
]

QUERY_COLUMNS = ["entity_id", "canonical_name", "query", "query_type", "candidate_reason"]

RESULT_COLUMNS = [
    "media_result_id", "entity_id", "canonical_name", "query", "query_type",
    "title", "snippet", "url", "source", "published_date", "retrieved_at",
    "candidate_reason",
]


class SearchResult(BaseModel):
    title: str
    snippet: str
    url: str
    source: str
    published_date: Optional[str] = None


class BaseSearchProvider(ABC):
    @abstractmethod
    def search(self, query: str, max_results: int) -> list[SearchResult]:
        """Search for a query and return normalized results."""


class MockSearchProvider(BaseSearchProvider):
    """Deterministic offline provider for pipeline development."""

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        results = []
        digest = hashlib.sha1(query.encode("utf-8")).hexdigest()
        for idx in range(max_results):
            token = digest[idx * 6:(idx + 1) * 6] or digest[:6]
            results.append(SearchResult(
                title=f"Mock result {idx + 1} for {query}",
                snippet="Deterministic placeholder result for offline adverse-media pipeline testing.",
                url=f"https://example.invalid/mock/{token}/{idx + 1}",
                source="mock",
                published_date=None,
            ))
        return results


def get_search_provider() -> BaseSearchProvider:
    if settings.search_provider != "mock" and not settings.search_api_key:
        logger.warning("SEARCH_PROVIDER=%s requested without SEARCH_API_KEY; falling back to mock.", settings.search_provider)
    return MockSearchProvider()


def select_media_candidates(
    entity: pd.DataFrame,
    zombie: pd.DataFrame,
    concentration: pd.DataFrame,
    market_shares: pd.DataFrame,
) -> pd.DataFrame:
    entity = ensure_columns(entity if entity is not None else pd.DataFrame(), CANDIDATE_COLUMNS)
    zombie = ensure_columns(zombie if zombie is not None else pd.DataFrame(), ["entity_id", "zombie_score"])
    market_shares = ensure_columns(market_shares if market_shares is not None else pd.DataFrame(), ["entity_id", "vendor_name", "vendor_share", "vendor_rank", "market_id"])
    concentration = ensure_columns(concentration if concentration is not None else pd.DataFrame(), ["market_id", "concentration_score", "concentration_risk_level"])

    base = entity.copy()
    if base.empty:
        return ensure_output_columns(pd.DataFrame(), CANDIDATE_COLUMNS)
    base["total_public_funding"] = pd.to_numeric(base["total_public_funding"], errors="coerce").fillna(0)
    base = base.merge(zombie[["entity_id", "zombie_score"]].drop_duplicates("entity_id"), on="entity_id", how="left", suffixes=("", "_z"))
    if "zombie_score_z" in base.columns:
        base["zombie_score"] = base["zombie_score"].fillna(base["zombie_score_z"])
        base = base.drop(columns=["zombie_score_z"])
    base["zombie_score"] = pd.to_numeric(base["zombie_score"], errors="coerce").fillna(0)

    reasons: dict[str, set[str]] = {}
    for row in base.sort_values("total_public_funding", ascending=False).head(100).itertuples(index=False):
        reasons.setdefault(row.entity_id, set()).add("top_100_total_public_funding")
    for row in base.sort_values("zombie_score", ascending=False).head(100).itertuples(index=False):
        reasons.setdefault(row.entity_id, set()).add("top_100_zombie_score")

    shares = market_shares.merge(concentration[["market_id", "concentration_score", "concentration_risk_level"]], on="market_id", how="left")
    vendor_metrics = pd.DataFrame()
    if not shares.empty:
        shares["vendor_share"] = pd.to_numeric(shares["vendor_share"], errors="coerce")
        top_vh = shares[(shares["concentration_risk_level"].eq("Very High")) & (pd.to_numeric(shares["vendor_rank"], errors="coerce") <= 3)]
        high_share = shares[shares["vendor_share"] >= 0.35]
        for frame, reason in ((top_vh, "top_3_vendor_very_high_concentration_market"), (high_share, "vendor_share_35pct_plus")):
            for row in frame.itertuples(index=False):
                if pd.notna(row.entity_id):
                    reasons.setdefault(row.entity_id, set()).add(reason)
        vendor_metrics = (
            shares.groupby("entity_id", dropna=False)
            .agg(
                canonical_name=("vendor_name", "first"),
                max_vendor_share=("vendor_share", "max"),
                max_concentration_score=("concentration_score", "max"),
            )
            .reset_index()
        )

    candidates = base[base["entity_id"].isin(reasons.keys())].copy()
    if not vendor_metrics.empty:
        missing_vendor_candidates = vendor_metrics[
            vendor_metrics["entity_id"].isin(reasons.keys())
            & ~vendor_metrics["entity_id"].isin(candidates["entity_id"])
        ].copy()
        if not missing_vendor_candidates.empty:
            candidates = pd.concat([
                candidates,
                pd.DataFrame({
                    "entity_id": missing_vendor_candidates["entity_id"],
                    "canonical_name": missing_vendor_candidates["canonical_name"],
                    "normalized_name": missing_vendor_candidates["canonical_name"].map(normalize_entity_name),
                    "source_systems": "AB_VENDOR",
                    "province": pd.NA,
                    "city": pd.NA,
                    "total_public_funding": 0,
                    "zombie_score": 0,
                }),
            ], ignore_index=True)
    candidates["candidate_reason"] = candidates["entity_id"].map(lambda e: ";".join(sorted(reasons.get(e, []))))
    if not vendor_metrics.empty:
        candidates = candidates.merge(vendor_metrics, on="entity_id", how="left", suffixes=("", "_vendor"))
        for col in ["max_vendor_share", "max_concentration_score"]:
            other = f"{col}_vendor"
            if other in candidates.columns:
                candidates[col] = candidates[col].fillna(candidates[other])
                candidates = candidates.drop(columns=[other])
    candidates["normalized_name"] = candidates["normalized_name"].fillna(candidates["canonical_name"].map(normalize_entity_name))
    candidates["_dedupe"] = candidates["entity_id"].fillna(candidates["normalized_name"])
    candidates = candidates.drop_duplicates("_dedupe").drop(columns=["_dedupe"])
    return ensure_output_columns(candidates, CANDIDATE_COLUMNS)


def generate_search_queries(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates is None or candidates.empty:
        return ensure_output_columns(pd.DataFrame(), QUERY_COLUMNS)
    rows: list[dict[str, Any]] = []
    base_terms = [
        ("fraud", "base"), ("investigation", "base"), ("lawsuit", "base"),
        ("fined", "base"), ("regulatory action", "base"), ("charged", "base"),
        ("audit", "base"), ("safety violation", "base"), ("sanctions", "base"),
        ("procurement", "base"),
    ]
    cra_terms = [("CRA revoked", "charity"), ("charity revoked", "charity"), ("misuse of funds", "charity"), ("fundraising complaint", "charity")]
    for row in candidates.itertuples(index=False):
        name = str(row.canonical_name)
        for term, qtype in base_terms:
            rows.append(_query_row(row, f'"{name}" {term}', qtype))
        source_systems = str(getattr(row, "source_systems", ""))
        is_cra = "CRA" in source_systems.upper()
        if is_cra:
            for term, qtype in cra_terms:
                rows.append(_query_row(row, f'"{name}" {term}', qtype))
        province = getattr(row, "province", None)
        if province is not None and not pd.isna(province) and str(province).strip():
            for term in ["fraud", "lawsuit", "investigation"]:
                rows.append(_query_row(row, f'"{name}" "{province}" {term}', "province"))
    return ensure_output_columns(pd.DataFrame(rows), QUERY_COLUMNS)


def run_media_search(queries: pd.DataFrame) -> pd.DataFrame:
    if queries is None or queries.empty:
        return ensure_output_columns(pd.DataFrame(), RESULT_COLUMNS)
    provider = get_search_provider()
    retrieved_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for query in queries.itertuples(index=False):
        for result in provider.search(str(query.query), settings.search_max_results_per_query):
            rows.append({
                "media_result_id": stable_hash(query.entity_id, result.url or result.title),
                "entity_id": query.entity_id,
                "canonical_name": query.canonical_name,
                "query": query.query,
                "query_type": query.query_type,
                "title": result.title,
                "snippet": result.snippet,
                "url": result.url,
                "source": result.source,
                "published_date": result.published_date,
                "retrieved_at": retrieved_at,
                "candidate_reason": query.candidate_reason,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return ensure_output_columns(df, RESULT_COLUMNS)
    df["_dedupe"] = [
        f"{row.entity_id}|{row.url}" if str(row.url).strip() else f"{row.entity_id}|{normalize_entity_name(row.title)}"
        for row in df.itertuples(index=False)
    ]
    df = df.drop_duplicates("_dedupe").drop(columns=["_dedupe"])
    return ensure_output_columns(df, RESULT_COLUMNS)


def _query_row(row: Any, query: str, query_type: str) -> dict[str, Any]:
    return {
        "entity_id": row.entity_id,
        "canonical_name": row.canonical_name,
        "query": query,
        "query_type": query_type,
        "candidate_reason": row.candidate_reason,
    }
