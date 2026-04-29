"""LLM review-friendly explanations for prioritized entities."""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, List

import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from config.settings import settings
from src.utils import ensure_columns, ensure_output_columns

logger = logging.getLogger(__name__)

REVIEW_EXPLANATION_COLUMNS = [
    "entity_id",
    "canonical_name",
    "overall_risk_score",
    "risk_level",
    "review_summary",
    "main_risk_indicators",
    "evidence_gaps",
    "recommended_next_steps",
    "safe_public_wording",
]

INVESTIGATION_PLAN_COLUMNS = [
    "entity_id",
    "canonical_name",
    "overall_risk_score",
    "risk_level",
    "recommended_workflow",
    "documents_to_check",
    "questions_for_analyst",
    "cautionary_notes",
]


class ReviewExplanation(BaseModel):
    review_summary: str
    main_risk_indicators: List[str]
    evidence_gaps: List[str]
    recommended_next_steps: List[str]
    safe_public_wording: str


class InvestigationPlan(BaseModel):
    recommended_workflow: List[str]
    documents_to_check: List[str]
    questions_for_analyst: List[str]
    cautionary_notes: List[str]


class BaseReviewGenerator(ABC):
    @abstractmethod
    def generate_review(self, context: dict[str, Any]) -> ReviewExplanation:
        """Generate review-friendly rationale."""

    @abstractmethod
    def generate_plan(self, context: dict[str, Any]) -> InvestigationPlan:
        """Generate analyst investigation plan."""


class MockReviewGenerator(BaseReviewGenerator):
    def generate_review(self, context: dict[str, Any]) -> ReviewExplanation:
        indicators = _indicator_list(context)
        gaps = _evidence_gaps(context)
        name = context.get("canonical_name", "This entity")
        return ReviewExplanation(
            review_summary=(
                f"{name} is prioritized for review because it combines "
                f"{', '.join(indicators[:3]) if indicators else 'limited but notable screening indicators'}. "
                "This does not establish wrongdoing, but it supports analyst review."
            ),
            main_risk_indicators=indicators,
            evidence_gaps=gaps,
            recommended_next_steps=_default_next_steps(context),
            safe_public_wording=(
                "Flagged for further review due to multiple screening indicators. "
                "No conclusion of wrongdoing is made by this system."
            ),
        )

    def generate_plan(self, context: dict[str, Any]) -> InvestigationPlan:
        return InvestigationPlan(
            recommended_workflow=_default_next_steps(context),
            documents_to_check=[
                "Latest registry status",
                "Most recent CRA filing or financial return",
                "Grant and contract payment records",
                "Adverse media source relevance and publication date",
            ],
            questions_for_analyst=[
                "Does the media evidence clearly refer to this entity?",
                "Did funding occur after any inactive, cancelled, revoked, or dissolved status?",
                "Does vendor concentration reflect legitimate specialization or dependency risk?",
            ],
            cautionary_notes=[
                "Screening flags are not findings of wrongdoing.",
                "Confirm source relevance before escalating.",
                "Review negative grant amounts as possible adjustments rather than new funding.",
            ],
        )


class OpenAIReviewGenerator(BaseReviewGenerator):
    def __init__(self) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=settings.openai_api_key)

    def generate_review(self, context: dict[str, Any]) -> ReviewExplanation:
        response = self.client.responses.parse(
            model=settings.openai_model,
            input=[
                {"role": "system", "content": _review_system_prompt()},
                {
                    "role": "user",
                    "content": "Generate a review-friendly analyst rationale:\n"
                    + json.dumps(context, ensure_ascii=False, default=str),
                },
            ],
            text_format=ReviewExplanation,
        )
        return response.output_parsed

    def generate_plan(self, context: dict[str, Any]) -> InvestigationPlan:
        response = self.client.responses.parse(
            model=settings.openai_model,
            input=[
                {"role": "system", "content": _plan_system_prompt()},
                {
                    "role": "user",
                    "content": "Generate a recommended analyst workflow:\n"
                    + json.dumps(context, ensure_ascii=False, default=str),
                },
            ],
            text_format=InvestigationPlan,
        )
        return response.output_parsed


def get_review_generator() -> BaseReviewGenerator:
    key = settings.openai_api_key or ""
    use_openai = (
        settings.review_explanation_provider == "openai"
        or settings.investigation_plan_provider == "openai"
    )
    if use_openai and key.startswith("sk-"):
        return OpenAIReviewGenerator()
    if use_openai:
        logger.warning("OpenAI review provider requested without OPENAI_API_KEY; using mock.")
    return MockReviewGenerator()


def build_review_explanations(
    risk_scores: pd.DataFrame,
    zombies: pd.DataFrame,
    adverse_media: pd.DataFrame,
) -> pd.DataFrame:
    scores = _prepare_scores(risk_scores)
    if scores.empty:
        return ensure_output_columns(pd.DataFrame(), REVIEW_EXPLANATION_COLUMNS)
    generator = get_review_generator() if settings.review_explanation_provider == "openai" else MockReviewGenerator()
    work = _prioritized(scores, settings.review_explanation_limit)
    contexts = _contexts(work, zombies, adverse_media)
    rows = []
    for context in tqdm(contexts, desc="Generating review explanations"):
        try:
            generated = generator.generate_review(context)
        except Exception as exc:
            logger.warning("Review explanation failed for %s: %s", context.get("entity_id"), exc)
            generated = MockReviewGenerator().generate_review(context)
        rows.append({
            "entity_id": context["entity_id"],
            "canonical_name": context["canonical_name"],
            "overall_risk_score": context.get("overall_risk_score"),
            "risk_level": context.get("risk_level"),
            "review_summary": generated.review_summary,
            "main_risk_indicators": "; ".join(generated.main_risk_indicators),
            "evidence_gaps": "; ".join(generated.evidence_gaps),
            "recommended_next_steps": "; ".join(generated.recommended_next_steps),
            "safe_public_wording": generated.safe_public_wording,
        })
    return ensure_output_columns(pd.DataFrame(rows), REVIEW_EXPLANATION_COLUMNS)


def build_investigation_plans(
    risk_scores: pd.DataFrame,
    zombies: pd.DataFrame,
    adverse_media: pd.DataFrame,
) -> pd.DataFrame:
    scores = _prepare_scores(risk_scores)
    if scores.empty:
        return ensure_output_columns(pd.DataFrame(), INVESTIGATION_PLAN_COLUMNS)
    generator = get_review_generator() if settings.investigation_plan_provider == "openai" else MockReviewGenerator()
    work = _prioritized(scores, settings.investigation_plan_limit)
    contexts = _contexts(work, zombies, adverse_media)
    rows = []
    for context in tqdm(contexts, desc="Generating investigation plans"):
        try:
            generated = generator.generate_plan(context)
        except Exception as exc:
            logger.warning("Investigation plan failed for %s: %s", context.get("entity_id"), exc)
            generated = MockReviewGenerator().generate_plan(context)
        rows.append({
            "entity_id": context["entity_id"],
            "canonical_name": context["canonical_name"],
            "overall_risk_score": context.get("overall_risk_score"),
            "risk_level": context.get("risk_level"),
            "recommended_workflow": "; ".join(generated.recommended_workflow),
            "documents_to_check": "; ".join(generated.documents_to_check),
            "questions_for_analyst": "; ".join(generated.questions_for_analyst),
            "cautionary_notes": "; ".join(generated.cautionary_notes),
        })
    return ensure_output_columns(pd.DataFrame(rows), INVESTIGATION_PLAN_COLUMNS)


def _prepare_scores(risk_scores: pd.DataFrame) -> pd.DataFrame:
    scores = ensure_columns(
        risk_scores if risk_scores is not None else pd.DataFrame(),
        [
            "entity_id", "canonical_name", "entity_type", "total_public_funding",
            "overall_risk_score", "risk_level", "zombie_score", "max_vendor_share",
            "max_concentration_score",
        ],
    )
    if scores.empty:
        return scores
    scores = scores.copy()
    scores["entity_id"] = scores["entity_id"].astype(str)
    scores["overall_risk_score"] = pd.to_numeric(scores["overall_risk_score"], errors="coerce").fillna(0)
    return scores


def _prioritized(scores: pd.DataFrame, limit: int) -> pd.DataFrame:
    work = scores.sort_values("overall_risk_score", ascending=False)
    if limit and limit > 0:
        return work.head(limit).copy()
    return work.copy()


def _contexts(scores: pd.DataFrame, zombies: pd.DataFrame, adverse_media: pd.DataFrame) -> List[dict[str, Any]]:
    zombies = ensure_columns(zombies if zombies is not None else pd.DataFrame(), ["entity_id", "zombie_flags"])
    adverse_media = ensure_columns(
        adverse_media if adverse_media is not None else pd.DataFrame(),
        [
            "entity_id", "adverse_media_type", "severity_score", "confidence_score",
            "summary", "title", "url", "requires_manual_review",
        ],
    )
    zombies["entity_id"] = zombies["entity_id"].astype(str)
    adverse_media["entity_id"] = adverse_media["entity_id"].astype(str)
    contexts = []
    for row in scores.itertuples(index=False):
        entity_id = str(row.entity_id)
        z = zombies[zombies["entity_id"].eq(entity_id)].head(1)
        media = adverse_media[adverse_media["entity_id"].eq(entity_id)].copy()
        if not media.empty:
            media["severity_score"] = pd.to_numeric(media["severity_score"], errors="coerce").fillna(0)
            media = media.sort_values("severity_score", ascending=False).head(5)
        contexts.append({
            "entity_id": entity_id,
            "canonical_name": row.canonical_name,
            "entity_type": getattr(row, "entity_type", ""),
            "total_public_funding": getattr(row, "total_public_funding", 0),
            "overall_risk_score": getattr(row, "overall_risk_score", 0),
            "risk_level": getattr(row, "risk_level", ""),
            "zombie_score": getattr(row, "zombie_score", 0),
            "zombie_flags": z["zombie_flags"].iloc[0] if not z.empty else "",
            "vendor_concentration_score": getattr(row, "max_concentration_score", 0),
            "max_vendor_share": getattr(row, "max_vendor_share", 0),
            "adverse_media_events": media[
                ["adverse_media_type", "severity_score", "confidence_score", "summary", "title", "url"]
            ].to_dict("records") if not media.empty else [],
        })
    return contexts


def _indicator_list(context: dict[str, Any]) -> List[str]:
    indicators = []
    if float(context.get("total_public_funding") or 0) >= 1_000_000:
        indicators.append("high public funding exposure")
    if float(context.get("zombie_score") or 0) >= 50:
        indicators.append("zombie-recipient screening indicators")
    if float(context.get("vendor_concentration_score") or 0) >= 50:
        indicators.append("vendor concentration indicators")
    if float(context.get("max_vendor_share") or 0) >= 0.35:
        indicators.append("large vendor market share")
    if context.get("adverse_media_events"):
        indicators.append("candidate adverse-media results requiring confirmation")
    return indicators or ["limited currently observed risk indicators"]


def _evidence_gaps(context: dict[str, Any]) -> List[str]:
    gaps = []
    if not context.get("adverse_media_events"):
        gaps.append("No confirmed adverse-media event is attached in the current output.")
    gaps.append("Registry and filing status should be verified against source records.")
    gaps.append("Media relevance should be confirmed before escalation.")
    return gaps


def _default_next_steps(context: dict[str, Any]) -> List[str]:
    return [
        "Verify current registry status.",
        "Confirm latest CRA filing year and filing completeness.",
        "Review grant, contract, and payment dates.",
        "Check whether funding occurred after inactive, cancelled, revoked, or dissolved status.",
        "Validate adverse-media source relevance and entity match.",
        "Assess whether vendor concentration reflects legitimate specialization or dependency risk.",
    ]


def _review_system_prompt() -> str:
    return (
        "You are an analyst assistant for a public funding risk screening system. "
        "Turn structured signals into concise review rationale. Do not say or imply fraud, crime, "
        "or wrongdoing is established. Use careful wording such as 'prioritized for review', "
        "'screening indicators', 'available data suggests', and 'requires confirmation'."
    )


def _plan_system_prompt() -> str:
    return (
        "You generate analyst investigation plans from structured screening signals. "
        "Recommend verification steps only. Do not make legal conclusions or accusatory claims."
    )
