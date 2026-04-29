"""LLM-assisted adverse media classification.

The classifier produces screening signals only. It must not assert that an
entity committed fraud, crime, or wrongdoing.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import pandas as pd
from pydantic import BaseModel, Field
from tqdm import tqdm

from config.settings import settings
from src.utils import ensure_columns, ensure_output_columns, stable_hash

logger = logging.getLogger(__name__)

ADVERSE_MEDIA_TYPES = [
    "regulatory_enforcement",
    "criminal_case",
    "civil_lawsuit",
    "fraud_or_misrepresentation",
    "procurement_issue",
    "audit_noncompliance",
    "charity_revocation",
    "safety_incident",
    "sanctions",
    "political_noise",
    "unrelated",
    "uncertain",
]

ADVERSE_MEDIA_EVENT_COLUMNS = [
    "media_result_id",
    "entity_id",
    "canonical_name",
    "title",
    "url",
    "source",
    "published_date",
    "is_relevant_entity",
    "entity_match_confidence",
    "is_adverse_media",
    "adverse_media_type",
    "severity_score",
    "confidence_score",
    "source_credibility_score",
    "summary",
    "evidence_text",
    "why_flagged",
    "why_not_flagged",
    "requires_manual_review",
]


class AdverseMediaClassification(BaseModel):
    is_relevant_entity: bool
    entity_match_confidence: float = Field(ge=0, le=1)
    is_adverse_media: bool
    adverse_media_type: str
    severity_score: int = Field(ge=0, le=100)
    confidence_score: float = Field(ge=0, le=1)
    source_credibility_score: float = Field(ge=0, le=1)
    summary: str
    evidence_text: str
    why_flagged: str
    why_not_flagged: str
    requires_manual_review: bool


class BaseAdverseMediaClassifier(ABC):
    @abstractmethod
    def classify(self, row: pd.Series) -> AdverseMediaClassification:
        """Classify one media search result."""


class MockAdverseMediaClassifier(BaseAdverseMediaClassifier):
    """Deterministic classifier for offline demos and tests."""

    def classify(self, row: pd.Series) -> AdverseMediaClassification:
        text = " ".join(str(row.get(c, "")) for c in ["query", "title", "snippet"]).lower()
        label = "uncertain"
        severity = 15
        if "cra revoked" in text or "charity revoked" in text:
            label, severity = "charity_revocation", 70
        elif "sanction" in text:
            label, severity = "sanctions", 75
        elif "charged" in text:
            label, severity = "criminal_case", 80
        elif "lawsuit" in text:
            label, severity = "civil_lawsuit", 55
        elif "fraud" in text or "misrepresentation" in text:
            label, severity = "fraud_or_misrepresentation", 75
        elif "regulatory action" in text or "fined" in text:
            label, severity = "regulatory_enforcement", 60
        elif "audit" in text:
            label, severity = "audit_noncompliance", 50
        elif "procurement" in text:
            label, severity = "procurement_issue", 50
        elif "safety violation" in text:
            label, severity = "safety_incident", 55

        is_mock = str(row.get("source", "")).lower() == "mock"
        is_adverse = label not in {"uncertain", "unrelated"}
        confidence = 0.35 if is_mock else 0.55
        return AdverseMediaClassification(
            is_relevant_entity=True,
            entity_match_confidence=0.65 if is_mock else 0.75,
            is_adverse_media=is_adverse,
            adverse_media_type=label,
            severity_score=severity if is_adverse else 10,
            confidence_score=confidence,
            source_credibility_score=0.2 if is_mock else 0.6,
            summary="Deterministic classification based on search query and snippet text.",
            evidence_text=str(row.get("snippet", ""))[:500],
            why_flagged=(
                "Flagged for further review due to adverse-media keywords in the search result."
                if is_adverse else ""
            ),
            why_not_flagged=(
                "" if is_adverse else "Insufficient evidence of serious adverse media in available text."
            ),
            requires_manual_review=is_adverse or label == "uncertain",
        )


class OpenAIAdverseMediaClassifier(BaseAdverseMediaClassifier):
    """OpenAI Structured Outputs classifier."""

    def __init__(self) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=settings.openai_api_key)

    def classify(self, row: pd.Series) -> AdverseMediaClassification:
        prompt = {
            "entity": str(row.get("canonical_name", "")),
            "query": str(row.get("query", "")),
            "title": str(row.get("title", "")),
            "snippet": str(row.get("snippet", "")),
            "url": str(row.get("url", "")),
            "source": str(row.get("source", "")),
            "published_date": str(row.get("published_date", "")),
            "allowed_adverse_media_types": ADVERSE_MEDIA_TYPES,
        }
        response = self.client.responses.parse(
            model=settings.openai_model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You classify media search results for a public funding risk screening workflow. "
                        "Return structured fields only. Do not conclude or imply wrongdoing. "
                        "Use 'unrelated' when the result is not about the entity. "
                        "Use 'political_noise' for partisan commentary without concrete adverse evidence. "
                        "Use 'uncertain' when evidence is too thin. "
                        "Serious adverse media requires concrete evidence in the title/snippet, such as enforcement, "
                        "charges, lawsuits, sanctions, audit findings, revocation, procurement issues, or safety incidents."
                    ),
                },
                {
                    "role": "user",
                    "content": "Classify this search result as JSON-compatible structured data:\n"
                    + json.dumps(prompt, ensure_ascii=False),
                },
            ],
            text_format=AdverseMediaClassification,
        )
        parsed = response.output_parsed
        if parsed.adverse_media_type not in ADVERSE_MEDIA_TYPES:
            parsed.adverse_media_type = "uncertain"
            parsed.requires_manual_review = True
        return parsed


def get_classifier() -> BaseAdverseMediaClassifier:
    key = settings.openai_api_key or ""
    has_real_key = key.startswith("sk-")
    if settings.adverse_media_classifier == "openai":
        if has_real_key:
            return OpenAIAdverseMediaClassifier()
        logger.warning("ADVERSE_MEDIA_CLASSIFIER=openai but OPENAI_API_KEY is missing; using mock classifier.")
    return MockAdverseMediaClassifier()


def classify_media_results(results: pd.DataFrame) -> pd.DataFrame:
    results = ensure_columns(
        results if results is not None else pd.DataFrame(),
        [
            "media_result_id", "entity_id", "canonical_name", "query", "title",
            "snippet", "url", "source", "published_date",
        ],
    )
    if results.empty:
        return ensure_output_columns(pd.DataFrame(), ADVERSE_MEDIA_EVENT_COLUMNS)

    classifier = get_classifier()
    work = results.copy()
    work = _attach_entity_matches(work)
    if isinstance(classifier, OpenAIAdverseMediaClassifier) and settings.adverse_media_classification_limit > 0:
        work = work.head(settings.adverse_media_classification_limit).copy()
        logger.warning(
            "OpenAI classifier enabled; limiting classification to %d rows. Set ADVERSE_MEDIA_CLASSIFICATION_LIMIT=0 for all rows.",
            settings.adverse_media_classification_limit,
        )

    rows: List[dict[str, Any]] = []
    for _, row in tqdm(work.iterrows(), total=len(work), desc="Classifying media results"):
        try:
            if _is_known_false_positive(row):
                classification = _false_positive_classification(row)
            else:
                classification = classifier.classify(row)
        except Exception as exc:
            logger.warning("Classification failed for %s: %s", row.get("media_result_id"), exc)
            classification = AdverseMediaClassification(
                is_relevant_entity=False,
                entity_match_confidence=0,
                is_adverse_media=False,
                adverse_media_type="uncertain",
                severity_score=0,
                confidence_score=0,
                source_credibility_score=0,
                summary="Classification failed; manual review required if this result is material.",
                evidence_text=str(row.get("snippet", ""))[:500],
                why_flagged="",
                why_not_flagged="Classifier failure or insufficient evidence.",
                requires_manual_review=True,
            )
        event = classification.model_dump()
        event.update({
            "media_result_id": row.get("media_result_id") or stable_hash(row.get("entity_id"), row.get("url"), row.get("title")),
            "entity_id": row.get("entity_id"),
            "canonical_name": row.get("canonical_name"),
            "title": row.get("title"),
            "url": row.get("url"),
            "source": row.get("source"),
            "published_date": row.get("published_date"),
        })
        rows.append(event)

    events = pd.DataFrame(rows)
    return ensure_output_columns(events, ADVERSE_MEDIA_EVENT_COLUMNS)


def _attach_entity_matches(results: pd.DataFrame) -> pd.DataFrame:
    from src.utils import read_csv_safe

    matches = read_csv_safe("media_entity_matches.csv")
    if matches is None or matches.empty or "media_result_id" not in matches.columns:
        return results
    keep = [
        "media_result_id", "is_same_entity", "match_confidence", "match_level",
        "why_same_entity", "why_not_same_entity", "false_positive_reason",
    ]
    keep = [c for c in keep if c in matches.columns]
    merged = results.merge(matches[keep].drop_duplicates("media_result_id"), on="media_result_id", how="left")
    logger.info("Attached %d entity disambiguation rows to media classification.", merged["is_same_entity"].notna().sum())
    return merged


def _is_known_false_positive(row: pd.Series) -> bool:
    if "is_same_entity" not in row or pd.isna(row.get("is_same_entity")):
        return False
    same = str(row.get("is_same_entity")).strip().lower() in {"true", "1", "yes"}
    confidence = pd.to_numeric(row.get("match_confidence"), errors="coerce")
    level = str(row.get("match_level", "")).lower()
    return (not same and pd.notna(confidence) and confidence <= 0.35) or level == "unrelated"


def _false_positive_classification(row: pd.Series) -> AdverseMediaClassification:
    why_not = str(row.get("why_not_same_entity") or row.get("false_positive_reason") or "")
    match_confidence = pd.to_numeric(row.get("match_confidence"), errors="coerce")
    if pd.isna(match_confidence):
        match_confidence = 0.0
    return AdverseMediaClassification(
        is_relevant_entity=False,
        entity_match_confidence=float(match_confidence),
        is_adverse_media=False,
        adverse_media_type="unrelated",
        severity_score=0,
        confidence_score=0.8,
        source_credibility_score=0.0 if str(row.get("source", "")).lower() == "mock" else 0.5,
        summary="Filtered as a likely entity false positive before adverse-media classification.",
        evidence_text=str(row.get("snippet", ""))[:500],
        why_flagged="",
        why_not_flagged=why_not or "Entity disambiguation indicates this result is likely not the same organization.",
        requires_manual_review=False,
    )
