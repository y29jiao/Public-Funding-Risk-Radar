"""Step 10: generate review-friendly LLM analyst rationales."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.review_explanations import build_review_explanations
from src.utils import configure_logging, read_csv_safe, write_csv


def main() -> None:
    configure_logging()
    explanations = build_review_explanations(
        read_csv_safe("entity_risk_scores.csv"),
        read_csv_safe("zombie_signals.csv"),
        read_csv_safe("adverse_media_events.csv"),
    )
    path = write_csv(explanations, "entity_review_explanations.csv")
    print(f"Wrote {len(explanations)} rows to {path}")


if __name__ == "__main__":
    main()
