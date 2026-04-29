"""LLM entity disambiguation for media false-positive filtering.

This step does not make legal conclusions. It only estimates whether a media
result appears to refer to the same entity as the database record.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, List

import pandas as pd
from pydantic import BaseModel, Field
from tqdm import tqdm

from config.settings import settings
from src.entity_resolution import normalize_entity_name
from src.utils import ensure_columns, ensure_output_columns, stable_hash

logger = logging.getLogger(__name__)

ENTITY_MATCH_COLUMNS = [
    "media_result_id",
    "entity_id",
    "canonical_name",
    "province",
    "city",
    "business_number",
    "charity_number",
    "source_systems",
    "title",
    "url",
    "source",
    "published_date",
    "is_same_entity",
    "match_confidence",
    "match_level",
    "evidence_text",
    "why_same_entity",
    "why_not_same_entity",
    "false_positive_reason",
    "requires_manual_review",
]

MATCH_LEVELS = ["strong", "probable", "possible", "unlikely", "unrelated", "uncertain"]


class EntityDisambiguationResult(BaseModel):
    is_same_entity: bool
    match_confidence: float = Field(ge=0, le=1)
    match_level: str
    evidence_text: str
    why_same_entity: str
    why_not_same_entity: str
    false_positive_reason: str
    requires_manual_review: bool


class BaseEntityDisambiguator(ABC):
    @abstractmethod
    def disambiguate(self, row: pd.Series) -> EntityDisambiguationResult:
        """Decide whether a media result appears to refer to the same entity."""


class MockEntityDisambiguator(BaseEntityDisambiguator):
    """Deterministic offline disambiguator."""

    def disambiguate(self, row: pd.Series) -> EntityDisambiguationResult:
        entity = normalize_entity_name(row.get("canonical_name", ""))
        title = normalize_entity_name(row.get("title", ""))
        snippet = normalize_entity_name(row.get("snippet", ""))
        text = f"{title} {snippet}"
        entity_tokens = [t for t in entity.split() if len(t) > 2]
        overlap = sum(1 for token in entity_tokens if token in text)
        ratio = overlap / max(len(entity_tokens), 1)
        province = str(row.get("province", "")).lower()
        city = str(row.get("city", "")).lower()
        geo_bonus = 0.1 if province and province != "nan" and province in text else 0
        geo_bonus += 0.1 if city and city != "nan" and city in text else 0
        confidence = min(0.95, ratio + geo_bonus)
        is_same = confidence >= 0.45 or str(row.get("source", "")).lower() == "mock"
        if confidence >= 0.75:
            level = "strong"
        elif confidence >= 0.55:
            level = "probable"
        elif confidence >= 0.35:
            level = "possible"
        elif confidence >= 0.15:
            level = "unlikely"
        else:
            level = "unrelated"
        if str(row.get("source", "")).lower() == "mock":
            confidence = max(confidence, 0.65)
            level = "probable"
            is_same = True
        return EntityDisambiguationResult(
            is_same_entity=is_same,
            match_confidence=round(float(confidence), 3),
            match_level=level,
            evidence_text=str(row.get("snippet", ""))[:500],
            why_same_entity="Name tokens and available geography appear consistent." if is_same else "",
            why_not_same_entity="" if is_same else "Available title/snippet do not clearly match the target entity.",
            false_positive_reason="" if is_same else "possible_name_collision_or_unrelated_result",
            requires_manual_review=confidence < 0.75,
        )


class OpenAIEntityDisambiguator(BaseEntityDisambiguator):
    """OpenAI Structured Outputs entity disambiguator."""

    def __init__(self) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=settings.openai_api_key)

    def disambiguate(self, row: pd.Series) -> EntityDisambiguationResult:
        prompt = {
            "database_entity": {
                "canonical_name": str(row.get("canonical_name", "")),
                "province": str(row.get("province", "")),
                "city": str(row.get("city", "")),
                "business_number": str(row.get("business_number", "")),
                "charity_number": str(row.get("charity_number", "")),
                "source_systems": str(row.get("source_systems", "")),
            },
            "media_result": {
                "title": str(row.get("title", "")),
                "snippet": str(row.get("snippet", "")),
                "url": str(row.get("url", "")),
                "source": str(row.get("source", "")),
                "published_date": str(row.get("published_date", "")),
            },
            "allowed_match_levels": MATCH_LEVELS,
        }
        response = self.client.responses.parse(
            model=settings.openai_model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You reduce false positives in a public funding media review workflow. "
                        "Your only task is entity disambiguation: decide whether the media result "
                        "appears to refer to the same organization as the database entity. "
                        "Do not classify wrongdoing and do not make legal conclusions. "
                        "Use geography, business/charity numbers, exact names, suffixes, and context. "
                        "If evidence is thin or ambiguous, choose uncertain/possible and require manual review."
                    ),
                },
                {
                    "role": "user",
                    "content": "Disambiguate this entity/media pair:\n" + json.dumps(prompt, ensure_ascii=False),
                },
            ],
            text_format=EntityDisambiguationResult,
        )
        parsed = response.output_parsed
        if parsed.match_level not in MATCH_LEVELS:
            parsed.match_level = "uncertain"
            parsed.requires_manual_review = True
        return parsed


def get_disambiguator() -> BaseEntityDisambiguator:
    key = settings.openai_api_key or ""
    if settings.entity_disambiguation_classifier == "openai":
        if key.startswith("sk-"):
            return OpenAIEntityDisambiguator()
        logger.warning("ENTITY_DISAMBIGUATION_CLASSIFIER=openai but OPENAI_API_KEY is missing; using mock.")
    return MockEntityDisambiguator()


def build_entity_match_inputs(media_results: pd.DataFrame, entities: pd.DataFrame) -> pd.DataFrame:
    media_results = ensure_columns(
        media_results if media_results is not None else pd.DataFrame(),
        ["media_result_id", "entity_id", "canonical_name", "title", "snippet", "url", "source", "published_date"],
    )
    entities = ensure_columns(
        entities if entities is not None else pd.DataFrame(),
        ["entity_id", "province", "city", "business_number", "charity_number", "source_systems"],
    )
    if media_results.empty:
        return media_results
    media = media_results.copy()
    media["entity_id"] = media["entity_id"].astype(str)
    entities = entities.copy()
    entities["entity_id"] = entities["entity_id"].astype(str)
    meta_cols = ["entity_id", "province", "city", "business_number", "charity_number", "source_systems"]
    return media.merge(entities[meta_cols].drop_duplicates("entity_id"), on="entity_id", how="left")


def disambiguate_media_entities(media_results: pd.DataFrame, entities: pd.DataFrame) -> pd.DataFrame:
    inputs = build_entity_match_inputs(media_results, entities)
    if inputs.empty:
        return ensure_output_columns(pd.DataFrame(), ENTITY_MATCH_COLUMNS)

    disambiguator = get_disambiguator()
    work = inputs.copy()
    if isinstance(disambiguator, OpenAIEntityDisambiguator) and settings.entity_disambiguation_limit > 0:
        work = work.head(settings.entity_disambiguation_limit).copy()
        logger.warning(
            "OpenAI entity disambiguation enabled; limiting to %d rows. Set ENTITY_DISAMBIGUATION_LIMIT=0 for all rows.",
            settings.entity_disambiguation_limit,
        )

    rows: List[dict[str, Any]] = []
    for _, row in tqdm(work.iterrows(), total=len(work), desc="Disambiguating media entities"):
        try:
            result = disambiguator.disambiguate(row)
        except Exception as exc:
            logger.warning("Entity disambiguation failed for %s: %s", row.get("media_result_id"), exc)
            result = EntityDisambiguationResult(
                is_same_entity=False,
                match_confidence=0,
                match_level="uncertain",
                evidence_text=str(row.get("snippet", ""))[:500],
                why_same_entity="",
                why_not_same_entity="Classifier failure or insufficient evidence.",
                false_positive_reason="classifier_failure_or_uncertain_match",
                requires_manual_review=True,
            )
        item = result.model_dump()
        item.update({
            "media_result_id": row.get("media_result_id") or stable_hash(row.get("entity_id"), row.get("url"), row.get("title")),
            "entity_id": row.get("entity_id"),
            "canonical_name": row.get("canonical_name"),
            "province": row.get("province"),
            "city": row.get("city"),
            "business_number": row.get("business_number"),
            "charity_number": row.get("charity_number"),
            "source_systems": row.get("source_systems"),
            "title": row.get("title"),
            "url": row.get("url"),
            "source": row.get("source"),
            "published_date": row.get("published_date"),
        })
        rows.append(item)
    return ensure_output_columns(pd.DataFrame(rows), ENTITY_MATCH_COLUMNS)
