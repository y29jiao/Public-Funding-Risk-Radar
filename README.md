# Public Funding Risk Radar

AI-assisted public funding risk review system for Canadian government and charity datasets.

The project unifies CRA charity filings, federal grants and contributions, Alberta grants/contracts/sole-source records, and a golden entity layer into a review workflow for analysts. It does not make legal conclusions. It flags entities for further review based on structured risk indicators, entity matching, and review-safe AI summaries.

## What It Does

Public Funding Risk Radar combines three screening lenses:

1. **Zombie Recipients**  
   Flags entities with indicators such as inactive/cancelled/revoked status, stopped filing after funding, stale CRA filings, high public funding dependency, or large funding exposure.

2. **Vendor Concentration**  
   Measures concentration in Alberta contract and sole-source markets using top vendor share, top-five share, HHI, and concentration risk levels.

3. **Adverse Media Review**  
   Builds media-search candidates, filters likely entity false positives, classifies media results into structured review categories, and produces analyst-friendly explanations.

## Why It Matters

Most public funding review workflows struggle with three problems:

- The same organization appears under different names across datasets.
- Search results create many false positives because unrelated entities share similar names.
- Raw risk indicators are hard to explain quickly to decision-makers.

This project addresses those problems by combining a fixed golden entity layer, risk scoring, LLM entity disambiguation, LLM adverse-media classification, and generated analyst workflows.

## Demo UI

The Streamlit UI is designed for reviewers who only open the webpage. They do not need to run scripts.

Main pages:

- `Welcome` explains the project and how to read the dashboard.
- `Executive Summary` shows output coverage, risk distribution, and review-ready highlights.
- `Review Queue` shows prioritized entities for analyst review.
- `AI Analyst Reports` shows generated rationale, next steps, and adverse-media summaries.
- `Evidence Tables` groups detailed zombie, concentration, entity-match, media, and quality-report outputs.
- `Ask the Risk Radar` provides lightweight Q&A grounded in generated CSV outputs.

Local launch:

```bash
streamlit run app.py
```

The current demo can also be exposed through ngrok when needed.

## Safety Framing

Use this wording in presentations:

> The LLM is not used to make legal conclusions. It is used to reduce false positives by matching media evidence to the correct entity and classifying whether the evidence is relevant for analyst review.

The system uses language such as:

- `flagged for further review`
- `screening indicators`
- `requires confirmation`
- `recommended analyst workflow`

It does not assert fraud, criminality, or wrongdoing.

## Current Outputs

| Output | Purpose |
| --- | --- |
| `outputs/entity_funding_summary.csv` | Entity-level funding, CRA, AB, FED, status, and dependency summary |
| `outputs/zombie_signals.csv` | Zombie-recipient screening flags and scores |
| `outputs/vendor_market_shares.csv` | Vendor-level market shares by source, department, category, and year |
| `outputs/vendor_concentration.csv` | Market-level HHI, top vendor share, and concentration risk |
| `outputs/adverse_media_candidates.csv` | Entities selected for media review |
| `outputs/media_search_queries.csv` | Generated search queries |
| `outputs/media_search_results.csv` | Mock or provider search results |
| `outputs/media_entity_matches.csv` | LLM/mock entity disambiguation and false-positive filtering |
| `outputs/adverse_media_events.csv` | LLM/mock adverse media classification |
| `outputs/entity_risk_scores.csv` | Composite entity risk score, risk level, priority, and explanation |
| `outputs/entity_review_explanations.csv` | LLM/mock review-friendly analyst rationale |
| `outputs/investigation_plans.csv` | LLM/mock recommended analyst workflows |
| `outputs/pipeline_quality_report.md` | Output validation report |

## Pipeline Steps

| Step | Script | Output |
| --- | --- | --- |
| 1 | `scripts/01_inspect_schema.py` | `schema_tables.csv`, `schema_columns.csv`, `schema_overview.md` |
| 2 | `scripts/02_build_entity_funding.py` | `entity_funding_summary.csv` |
| 3 | `scripts/03_compute_zombie_signals.py` | `zombie_signals.csv` |
| 4 | `scripts/04_compute_vendor_concentration.py` | `vendor_market_shares.csv`, `vendor_concentration.csv` |
| 5 | `scripts/05_search_adverse_media.py` | `adverse_media_candidates.csv`, `media_search_queries.csv`, `media_search_results.csv` |
| 6 | `scripts/06_build_risk_scores.py` | `entity_risk_scores.csv` |
| 7 | `scripts/07_validate_outputs.py` | `pipeline_quality_report.md` |
| 8 | `scripts/08_classify_adverse_media.py` | `adverse_media_events.csv` |
| 9 | `scripts/09_disambiguate_media_entities.py` | `media_entity_matches.csv` |
| 10 | `scripts/10_generate_review_explanations.py` | `entity_review_explanations.csv` |
| 11 | `scripts/11_generate_investigation_plans.py` | `investigation_plans.csv` |

Recommended internal run order after Step 5:

```bash
python scripts/06_build_risk_scores.py
python scripts/09_disambiguate_media_entities.py
python scripts/08_classify_adverse_media.py
python scripts/10_generate_review_explanations.py
python scripts/11_generate_investigation_plans.py
python scripts/07_validate_outputs.py
```

## Fixed-Table Mode

The pipeline uses the discovered schema directly instead of table-name inference:

- `general.entity_golden_records`
- `general.vw_entity_funding`
- `general.vw_entity_search`
- `fed.grants_contributions`
- `fed.vw_grants_decoded`
- `ab.ab_grants`
- `ab.ab_contracts`
- `ab.ab_sole_source`
- `ab.vw_non_profit_decoded`
- `cra.vw_charity_profiles`
- `cra.vw_charity_financials_by_year`
- `cra.govt_funding_by_charity`
- `cra.t3010_plausibility_flags`
- `cra.t3010_completeness_issues`
- `cra.t3010_impossibilities`

## OpenAI Usage

OpenAI is optional and bounded by limits in `.env`.

Current small-batch demo settings:

```bash
ADVERSE_MEDIA_CLASSIFIER=openai
ADVERSE_MEDIA_CLASSIFICATION_LIMIT=20
ENTITY_DISAMBIGUATION_CLASSIFIER=openai
ENTITY_DISAMBIGUATION_LIMIT=20
REVIEW_EXPLANATION_PROVIDER=openai
REVIEW_EXPLANATION_LIMIT=10
INVESTIGATION_PLAN_PROVIDER=openai
INVESTIGATION_PLAN_LIMIT=10
ANALYST_CHAT_PROVIDER=openai
OPENAI_MODEL=gpt-4o-mini
```

Set any provider/classifier to `mock` to run offline without token usage.

## Search Provider

The project currently uses:

```bash
SEARCH_PROVIDER=mock
```

This creates deterministic offline search results so the full pipeline can run during demos. The OpenAI components can still classify and summarize those rows. For production-like adverse media review, the next step is to connect a real search provider.

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with:

- `DATABASE_URL`
- `OPENAI_API_KEY` if using OpenAI features
- search provider settings if adding real search

## Validation

Run:

```bash
python scripts/07_validate_outputs.py
```

This writes:

```text
outputs/pipeline_quality_report.md
```

The report checks whether all expected output files exist and contain the required columns.

## Project Strengths

- Uses a real multi-source public funding database rather than toy data.
- Pushes aggregation into SQL for large-table safety.
- Preserves both `gross_positive_funding` and `net_funding` because AB grants can include negative adjustments.
- Uses entity disambiguation to reduce media false positives.
- Produces analyst-ready summaries and investigation plans, not just scores.
- Keeps legal and ethical boundaries explicit by using review-safe language.
