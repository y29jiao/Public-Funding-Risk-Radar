"""Streamlit dashboard for Public Funding Risk Radar."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from src.analyst_chat import answer_question, test_openai_connectivity

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"


@st.cache
def load_csv(name: str, nrows: Optional[int] = None) -> pd.DataFrame:
    path = OUTPUTS / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, nrows=nrows)


def main() -> None:
    st.set_page_config(page_title="Public Funding Risk Radar", layout="wide")
    page = st.sidebar.radio(
        "Risk Radar",
        [
            "Welcome",
            "Executive Summary",
            "Review Queue",
            "AI Agent Workflow",
            "AI Analyst Reports",
            "Evidence Tables",
            "Ask the Risk Radar",
        ],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Screening support only. No finding of wrongdoing.")

    st.title("Public Funding Risk Radar")
    st.caption(
        "AI-assisted screening for analyst review. Flags identify entities for further review and do not claim wrongdoing."
    )

    if page == "Welcome":
        welcome_page()
    elif page == "Executive Summary":
        executive_summary_page()
    elif page == "Review Queue":
        review_queue_page()
    elif page == "AI Agent Workflow":
        ai_agent_workflow_page()
    elif page == "AI Analyst Reports":
        ai_reports_page()
    elif page == "Evidence Tables":
        evidence_tables_page()
    else:
        ask_risk_radar_page()


def welcome_page() -> None:
    st.subheader("Project Overview")
    st.write(
        "Public Funding Risk Radar is an AI-assisted screening system for analyst review. "
        "It combines structured funding, filing, and vendor concentration signals with "
        "adverse-media workflows to prioritize entities for further review."
    )
    st.write(
        "The system is designed to support triage, not to make legal conclusions. "
        "All flags are review signals and do not establish wrongdoing."
    )
    st.caption(
        "For process design: see the **AI Agent Workflow** page for a concise view of red-tape reduction and human-in-the-loop review."
    )

    st.subheader("How To Use")
    st.markdown("1. Open **Executive Summary** to see current output coverage and risk distribution.")
    st.markdown("2. Open **Review Queue** to inspect top-priority entities.")
    st.markdown("3. Open **AI Agent Workflow** to see how the system reduces red tape.")
    st.markdown("4. Open **AI Analyst Reports** for generated rationale and investigation plans.")
    st.markdown("5. Open **Evidence Tables** for detailed zombie, concentration, entity-match, and media evidence.")
    st.markdown("6. Use **Ask the Risk Radar** for guided Q&A grounded in generated outputs.")

    st.subheader("Reading Flow")
    st.write(
        "For reviewers, this dashboard is intended to be used directly without running commands. "
        "Start from Executive Summary, then Review Queue, then AI Agent Workflow, then AI Analyst Reports, "
        "and use Evidence Tables only when you need source-level details."
    )


def executive_summary_page() -> None:
    scores = load_csv("entity_risk_scores.csv")
    explanations = load_csv("entity_review_explanations.csv")
    quality = _quality_counts()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scored Entities", _count_label(quality, "entity_risk_scores.csv", len(scores)))
    c2.metric("Review Rationales", _count_label(quality, "entity_review_explanations.csv", len(explanations)))
    c3.metric("Media Events", _count_label(quality, "adverse_media_events.csv", 0))
    c4.metric("Vendor Markets", _count_label(quality, "vendor_concentration.csv", 0))

    if not scores.empty and "risk_level" in scores.columns:
        dist = _risk_distribution(scores)
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Low", f"{int(dist.loc['Low', 'count']):,}")
        h2.metric("Medium", f"{int(dist.loc['Medium', 'count']):,}")
        h3.metric("High", f"{int(dist.loc['High', 'count']):,}")
        h4.metric("Very High", f"{int(dist.loc['Very High', 'count']):,}")

    st.subheader("Most Review-Ready Outputs")
    if not explanations.empty:
        for row in explanations.sort_values("overall_risk_score", ascending=False).head(5).itertuples(index=False):
            with st.expander(f"{row.canonical_name} | {row.risk_level} | score {row.overall_risk_score}", expanded=False):
                st.write(row.review_summary)
                st.markdown("**Main indicators**")
                st.write(row.main_risk_indicators)
                st.markdown("**Safe wording**")
                st.write(row.safe_public_wording)
    else:
        st.info("Run scripts/10_generate_review_explanations.py to populate review-ready summaries.")

    st.subheader("Risk Level Distribution")
    if not scores.empty and "risk_level" in scores.columns:
        dist = _risk_distribution(scores)
        st.caption(
            "The distribution is highly skewed: most records are low-risk screening rows, "
            "so a normal bar chart makes the smaller high-risk queue hard to see."
        )
        st.table(dist[["count", "percent"]])
        st.markdown("**Log-scaled count view**")
        st.bar_chart(dist["log10_count"])
    else:
        st.info("Run scripts/06_build_risk_scores.py to populate risk scores.")


def review_queue_page() -> None:
    scores = load_csv("entity_risk_scores.csv")
    explanations = load_csv("entity_review_explanations.csv")
    plans = load_csv("investigation_plans.csv")
    if scores.empty:
        st.info("Run scripts/06_build_risk_scores.py to generate entity_risk_scores.csv.")
        return

    st.subheader("Prioritized Review Queue")
    top = scores.sort_values("overall_risk_score", ascending=False).head(50).copy()
    cols = [
        "canonical_name", "overall_risk_score", "risk_level", "review_priority",
        "total_public_funding", "zombie_score", "max_vendor_share", "max_concentration_score",
    ]
    st.dataframe(top[[c for c in cols if c in top.columns]])

    st.markdown("### Review Rationales")
    _show_rationales(explanations)

    st.markdown("### Recommended Investigation Workflows")
    _show_plans(plans)


def ai_agent_workflow_page() -> None:
    st.subheader("AI Agent Workflow")
    st.caption(
        "The goal is to replace a rigid, document-heavy review path with a dynamic AI-assisted workflow "
        "where analysts keep final judgment."
    )

    st.markdown("### Red-Tape Process Replaced")
    st.table(pd.DataFrame([
        {
            "Existing bureaucratic step": "Analyst manually searches separate charity, grant, contract, and media sources.",
            "AI agent workflow": "Entity, funding, filing, vendor, and media signals are assembled into one review queue.",
            "Improvement": "Less handoff friction and fewer repeated lookups.",
        },
        {
            "Existing bureaucratic step": "Teams wait for full case files before deciding what deserves attention.",
            "AI agent workflow": "Risk scoring prioritizes the highest-signal entities first, then opens evidence only when needed.",
            "Improvement": "Earlier triage with a clearer review order.",
        },
        {
            "Existing bureaucratic step": "Name variants and unrelated search results create slow false-positive cleanup.",
            "AI agent workflow": "LLM entity disambiguation labels same-entity, possible-match, and false-positive media results.",
            "Improvement": "Analysts spend more time judging relevant evidence.",
        },
        {
            "Existing bureaucratic step": "Narratives and next steps are written manually from raw tables.",
            "AI agent workflow": "AI generates review-safe rationales, evidence gaps, and recommended investigation plans.",
            "Improvement": "Faster, more consistent case preparation.",
        },
    ]))

    st.markdown("### Dynamic Human-in-the-Loop Review")
    st.write(
        "The agent does not approve, deny, accuse, or make legal conclusions. It prepares a prioritized, "
        "explainable review package so a human analyst can confirm source records, weigh context, and decide "
        "whether escalation is justified."
    )
    st.table(pd.DataFrame([
        {"Stage": "1. Monitor", "Agent help": "Refresh structured outputs from public funding, filing, and vendor data.", "Human judgment": "Set policy thresholds and review priorities."},
        {"Stage": "2. Triage", "Agent help": "Rank entities by zombie, concentration, and adverse-media signals.", "Human judgment": "Choose which cases need attention now."},
        {"Stage": "3. Verify", "Agent help": "Filter likely media false positives and identify missing evidence.", "Human judgment": "Confirm registry records, funding dates, and source relevance."},
        {"Stage": "4. Explain", "Agent help": "Draft cautious summaries and investigation workflows.", "Human judgment": "Edit wording, add context, and decide next action."},
    ]))

def ai_reports_page() -> None:
    explanations = load_csv("entity_review_explanations.csv")
    plans = load_csv("investigation_plans.csv")
    adverse = load_csv("adverse_media_events.csv")

    st.subheader("AI Analyst Reports")
    st.caption("Narrative summaries are generated from structured signals and use review-safe language.")

    st.markdown("### Review-Friendly Rationales")
    _show_rationales(explanations)

    st.markdown("### Investigation Plans")
    _show_plans(plans)

    st.markdown("### Adverse Media Classifications")
    _show_adverse_media(adverse)


def evidence_tables_page() -> None:
    st.subheader("Evidence Tables")
    st.caption("Detailed analytic tables are grouped here to keep the demo flow readable.")

    with st.expander("Zombie Recipient Signals", expanded=False):
        zombie_page(load_csv("zombie_signals.csv"))

    with st.expander("Vendor Concentration", expanded=False):
        concentration_page(load_csv("vendor_concentration.csv"))

    with st.expander("Entity Match / False Positive Filter", expanded=False):
        entity_matches_page(load_csv("media_entity_matches.csv"))

    with st.expander("Adverse Media Events", expanded=False):
        _show_adverse_media(load_csv("adverse_media_events.csv"))

    with st.expander("Pipeline Quality Report", expanded=True):
        report = OUTPUTS / "pipeline_quality_report.md"
        st.markdown(report.read_text(encoding="utf-8") if report.exists() else "No quality report found.")


def ask_risk_radar_page() -> None:
    st.subheader("Ask the Risk Radar")
    st.caption("Answers are grounded in generated outputs only and use screening language.")
    question = st.text_input("Question", value="Why is the top entity prioritized for review?")
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Ask"):
            answer = answer_question(
                question,
                {
                    "scores": load_csv("entity_risk_scores.csv"),
                    "zombies": load_csv("zombie_signals.csv"),
                    "concentration": load_csv("vendor_concentration.csv"),
                    "adverse": load_csv("adverse_media_events.csv"),
                    "plans": load_csv("investigation_plans.csv"),
                },
            )
            st.write(answer)
    with c2:
        if st.button("Test OpenAI Connection"):
            status = test_openai_connectivity()
            if status.startswith("Connectivity OK"):
                st.success(status)
            else:
                st.warning(status)


def zombie_page(zombies: pd.DataFrame) -> None:
    if zombies.empty:
        st.info("Run scripts/03_compute_zombie_signals.py to generate zombie_signals.csv.")
        return
    cols = [
        "canonical_name", "zombie_score", "zombie_risk_level", "normalized_status",
        "ab_non_profit_status", "total_public_funding", "zombie_flags",
    ]
    show = zombies.sort_values("zombie_score", ascending=False).head(100)
    st.dataframe(show[[c for c in cols if c in show.columns]])


def concentration_page(concentration: pd.DataFrame) -> None:
    if concentration.empty:
        st.info("Run scripts/04_compute_vendor_concentration.py to generate vendor_concentration.csv.")
        return
    cols = [
        "source", "department", "category", "year", "total_spending",
        "vendor_count", "top_1_vendor", "top_1_share", "top_5_share",
        "hhi", "concentration_score", "concentration_risk_level",
    ]
    show = concentration.sort_values("concentration_score", ascending=False).head(100)
    st.dataframe(show[[c for c in cols if c in show.columns]])


def entity_matches_page(matches: pd.DataFrame) -> None:
    if matches.empty:
        st.info("Run scripts/09_disambiguate_media_entities.py to generate media_entity_matches.csv.")
        return
    cols = [
        "canonical_name", "is_same_entity", "match_confidence", "match_level",
        "requires_manual_review", "province", "city", "title",
        "why_same_entity", "why_not_same_entity", "false_positive_reason",
    ]
    show = matches.sort_values(["requires_manual_review", "match_confidence"], ascending=[False, True]).head(100)
    st.dataframe(show[[c for c in cols if c in show.columns]])


def _show_rationales(explanations: pd.DataFrame) -> None:
    if explanations.empty:
        st.info("Run scripts/10_generate_review_explanations.py to generate entity_review_explanations.csv.")
        return
    for row in explanations.sort_values("overall_risk_score", ascending=False).head(25).itertuples(index=False):
        with st.expander(f"{row.canonical_name} | {row.risk_level} | score {row.overall_risk_score}", expanded=False):
            st.write(row.review_summary)
            st.markdown("**Main risk indicators**")
            st.write(row.main_risk_indicators)
            st.markdown("**Evidence gaps**")
            st.write(row.evidence_gaps)
            st.markdown("**Recommended next steps**")
            st.write(row.recommended_next_steps)
            st.markdown("**Safe public wording**")
            st.write(row.safe_public_wording)


def _show_plans(plans: pd.DataFrame) -> None:
    if plans.empty:
        st.info("Run scripts/11_generate_investigation_plans.py to generate investigation_plans.csv.")
        return
    for row in plans.sort_values("overall_risk_score", ascending=False).head(25).itertuples(index=False):
        with st.expander(f"{row.canonical_name} | {row.risk_level} | score {row.overall_risk_score}", expanded=False):
            st.markdown("**Recommended workflow**")
            st.write(row.recommended_workflow)
            st.markdown("**Documents to check**")
            st.write(row.documents_to_check)
            st.markdown("**Questions for analyst**")
            st.write(row.questions_for_analyst)
            st.markdown("**Cautionary notes**")
            st.write(row.cautionary_notes)


def _show_adverse_media(events: pd.DataFrame) -> None:
    if events.empty:
        st.info("Run scripts/08_classify_adverse_media.py to generate adverse_media_events.csv.")
        return
    cols = [
        "canonical_name", "is_relevant_entity", "is_adverse_media",
        "adverse_media_type", "severity_score", "confidence_score",
        "requires_manual_review", "title", "summary", "why_flagged",
    ]
    show = events.sort_values(["severity_score", "confidence_score"], ascending=False).head(100)
    st.dataframe(show[[c for c in cols if c in show.columns]])


def _quality_counts() -> dict[str, int]:
    path = OUTPUTS / "pipeline_quality_report.md"
    if not path.exists():
        return {}
    counts = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            current = line.replace("## ", "").strip()
        elif current and line.startswith("Rows:"):
            try:
                counts[current] = int(line.split(":", 1)[1].strip())
            except ValueError:
                counts[current] = 0
    return counts


def _count_label(counts: dict[str, int], filename: str, fallback: int) -> str:
    return f"{counts.get(filename, fallback):,}"


def _risk_distribution(scores: pd.DataFrame) -> pd.DataFrame:
    order = ["Low", "Medium", "High", "Very High"]
    counts = scores["risk_level"].value_counts().reindex(order, fill_value=0)
    dist = pd.DataFrame({"count": counts})
    total = max(int(dist["count"].sum()), 1)
    dist["percent"] = (dist["count"] / total * 100).round(3).astype(str) + "%"
    dist["log10_count"] = (dist["count"] + 1).map(lambda v: __import__("math").log10(v))
    return dist


if __name__ == "__main__":
    main()
