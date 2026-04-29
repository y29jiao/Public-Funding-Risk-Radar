"""Step 7: validate generated CSV outputs and write a quality report."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.funding_pipeline import ENTITY_FUNDING_COLUMNS
from src.zombie_detector import ZOMBIE_COLUMNS
from src.concentration import CONCENTRATION_COLUMNS, MARKET_SHARE_COLUMNS
from src.media_search import CANDIDATE_COLUMNS, QUERY_COLUMNS, RESULT_COLUMNS
from src.risk_scoring import RISK_SCORE_COLUMNS
from src.adverse_media_classifier import ADVERSE_MEDIA_EVENT_COLUMNS
from src.entity_disambiguation import ENTITY_MATCH_COLUMNS
from src.review_explanations import REVIEW_EXPLANATION_COLUMNS, INVESTIGATION_PLAN_COLUMNS
from src.utils import configure_logging, output_path


EXPECTED = {
    "entity_funding_summary.csv": ENTITY_FUNDING_COLUMNS,
    "zombie_signals.csv": ZOMBIE_COLUMNS,
    "vendor_market_shares.csv": MARKET_SHARE_COLUMNS,
    "vendor_concentration.csv": CONCENTRATION_COLUMNS,
    "adverse_media_candidates.csv": CANDIDATE_COLUMNS,
    "media_search_queries.csv": QUERY_COLUMNS,
    "media_search_results.csv": RESULT_COLUMNS,
    "media_entity_matches.csv": ENTITY_MATCH_COLUMNS,
    "adverse_media_events.csv": ADVERSE_MEDIA_EVENT_COLUMNS,
    "entity_risk_scores.csv": RISK_SCORE_COLUMNS,
    "entity_review_explanations.csv": REVIEW_EXPLANATION_COLUMNS,
    "investigation_plans.csv": INVESTIGATION_PLAN_COLUMNS,
}


def main() -> None:
    configure_logging()
    lines = ["# Pipeline Quality Report", ""]
    for filename, columns in EXPECTED.items():
        path = output_path(filename)
        lines.extend([f"## {filename}", ""])
        if not path.exists():
            lines.extend(["Status: missing", ""])
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            lines.extend([f"Status: unreadable ({exc})", ""])
            continue
        missing = [c for c in columns if c not in df.columns]
        extra = [c for c in df.columns if c not in columns]
        lines.append(f"Rows: {len(df)}")
        lines.append(f"Columns: {len(df.columns)}")
        lines.append(f"Status: {'ok' if not missing else 'missing required columns'}")
        if missing:
            lines.append("Missing columns: " + ", ".join(missing))
        if extra:
            lines.append("Extra columns: " + ", ".join(extra))
        key_cols = [c for c in ["entity_id", "canonical_name", "overall_risk_score"] if c in df.columns]
        if key_cols and len(df) > 0:
            null_rates = df[key_cols].isna().mean().mul(100).round(1)
            lines.append("Key null rates: " + ", ".join(f"{k}={v}%" for k, v in null_rates.items()))
        lines.append("")

    report = output_path("pipeline_quality_report.md")
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
