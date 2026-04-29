"""Step 11: generate recommended analyst investigation plans."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.review_explanations import build_investigation_plans
from src.utils import configure_logging, read_csv_safe, write_csv


def main() -> None:
    configure_logging()
    plans = build_investigation_plans(
        read_csv_safe("entity_risk_scores.csv"),
        read_csv_safe("zombie_signals.csv"),
        read_csv_safe("adverse_media_events.csv"),
    )
    path = write_csv(plans, "investigation_plans.csv")
    print(f"Wrote {len(plans)} rows to {path}")


if __name__ == "__main__":
    main()
