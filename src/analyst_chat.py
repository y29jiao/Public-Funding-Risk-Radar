"""Lightweight analyst Q&A over generated pipeline outputs."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict

import pandas as pd

from config.settings import settings


def answer_question(question: str, datasets: Dict[str, pd.DataFrame]) -> str:
    """Answer using only generated CSV-derived context."""
    context = build_chat_context(question, datasets)
    key = settings.openai_api_key or ""
    if settings.analyst_chat_provider == "openai" and key.startswith("sk-"):
        last_exc = None
        for attempt in range(1, 4):
            try:
                return _openai_answer(question, context)
            except Exception as exc:
                last_exc = exc
                _append_chat_error(question, exc, attempt)
                if attempt < 3:
                    time.sleep(1.5 * attempt)
        return (
            "OpenAI chat is temporarily unavailable, so this is a fallback answer based on local outputs only. "
            + _mock_answer(question, context)
            + f" (fallback reason: {type(last_exc).__name__})"
        )
    return _mock_answer(question, context)


def build_chat_context(question: str, datasets: Dict[str, pd.DataFrame]) -> dict:
    scores = datasets.get("scores", pd.DataFrame()).copy()
    zombies = datasets.get("zombies", pd.DataFrame()).copy()
    concentration = datasets.get("concentration", pd.DataFrame()).copy()
    adverse = datasets.get("adverse", pd.DataFrame()).copy()
    plans = datasets.get("plans", pd.DataFrame()).copy()

    context = {
        "question": question,
        "top_risk_entities": _top_records(scores, "overall_risk_score", 10),
        "top_zombie_recipients": _top_records(zombies, "zombie_score", 10),
        "top_concentration_markets": _top_records(concentration, "concentration_score", 10),
        "top_adverse_media_events": _top_records(adverse, "severity_score", 10),
        "investigation_plans": _top_records(plans, "overall_risk_score", 5),
    }

    q = question.lower()
    if not scores.empty and "canonical_name" in scores.columns:
        matches = scores[scores["canonical_name"].astype(str).str.lower().map(_name_mentions_question(q))]
        if not matches.empty:
            entity_id = str(matches.iloc[0].get("entity_id"))
            context["matched_entity"] = matches.head(1).to_dict("records")
            context["matched_entity_zombie"] = zombies[zombies.get("entity_id", pd.Series(dtype=str)).astype(str).eq(entity_id)].head(3).to_dict("records") if not zombies.empty else []
            context["matched_entity_adverse_media"] = adverse[adverse.get("entity_id", pd.Series(dtype=str)).astype(str).eq(entity_id)].head(5).to_dict("records") if not adverse.empty else []
            context["matched_entity_plan"] = plans[plans.get("entity_id", pd.Series(dtype=str)).astype(str).eq(entity_id)].head(1).to_dict("records") if not plans.empty else []
    return context


def _top_records(df: pd.DataFrame, sort_col: str, n: int) -> list:
    if df is None or df.empty:
        return []
    work = df.copy()
    if sort_col in work.columns:
        work[sort_col] = pd.to_numeric(work[sort_col], errors="coerce").fillna(0)
        work = work.sort_values(sort_col, ascending=False)
    return work.head(n).to_dict("records")


def _openai_answer(question: str, context: dict) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key, timeout=45.0, max_retries=2)
    response = client.responses.create(
        model=settings.openai_model,
        input=[
            {
                "role": "system",
                "content": (
                    "You answer analyst questions using only the provided generated CSV context. "
                    "Do not claim fraud, criminality, or wrongdoing. Use screening language. "
                    "If the context is insufficient, say what output or source record should be checked next."
                ),
            },
            {
                "role": "user",
                "content": "Question: "
                + question
                + "\n\nContext JSON:\n"
                + json.dumps(context, ensure_ascii=False, default=str)[:50000],
            },
        ],
    )
    return response.output_text


def test_openai_connectivity() -> str:
    key = settings.openai_api_key or ""
    if not key.startswith("sk-"):
        return "OPENAI_API_KEY missing or invalid format."
    try:
        from openai import OpenAI

        client = OpenAI(api_key=key, timeout=20.0, max_retries=1)
        response = client.responses.create(
            model=settings.openai_model,
            input="Reply with exactly: OPENAI_OK",
        )
        return f"Connectivity OK ({response.output_text.strip()})"
    except Exception as exc:
        _append_chat_error("connectivity_test", exc, 1)
        return f"Connectivity failed: {type(exc).__name__}: {exc}"


def _mock_answer(question: str, context: dict) -> str:
    q = question.lower()
    if "zombie" in q:
        return "The top zombie-recipient rows are available in the context from zombie_signals.csv. Review latest filing year, normalized status, and stopped-filing-after-funding flags."
    if "vendor" in q or "sole-source" in q or "concentration" in q:
        return "Vendor concentration is based on AB contracts and sole-source markets. Review top_1_share, top_5_share, HHI, and whether concentration reflects legitimate specialization or dependency risk."
    if context.get("matched_entity"):
        entity = context["matched_entity"][0]
        return (
            f"{entity.get('canonical_name')} is prioritized for review with an overall risk score of "
            f"{entity.get('overall_risk_score')}. The available generated outputs should be reviewed for "
            "zombie-recipient indicators, vendor concentration indicators, and adverse-media relevance. "
            "This is a screening summary only and does not establish wrongdoing."
        )
    return "I can answer using generated outputs such as entity_risk_scores.csv, zombie_signals.csv, vendor_concentration.csv, adverse_media_events.csv, and investigation_plans.csv. Ask about a specific entity or risk signal."


def _name_mentions_question(question: str):
    def matcher(name: str) -> bool:
        clean = str(name).lower().strip()
        if len(clean) >= 8 and clean in question:
            return True
        tokens = [t for t in clean.replace("&", " ").split() if len(t) >= 4]
        return len(tokens) >= 2 and all(token in question for token in tokens[:2])

    return matcher


def _append_chat_error(question: str, exc: Exception, attempt: int) -> None:
    log_path = Path(__file__).resolve().parent.parent / "outputs" / "chat_errors.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fp:
        fp.write(
            f"attempt={attempt} error={type(exc).__name__} question={question[:200]!r} detail={str(exc)[:500]}\n"
        )
