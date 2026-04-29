"""Step 8: classify adverse media search results with LLM or mock fallback."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.adverse_media_classifier import classify_media_results
from src.utils import configure_logging, read_csv_safe, write_csv


def main() -> None:
    configure_logging()
    results = read_csv_safe("media_search_results.csv")
    if results is None:
        print("Missing outputs/media_search_results.csv; writing empty adverse_media_events.csv")
    events = classify_media_results(results)
    path = write_csv(events, "adverse_media_events.csv")
    print(f"Wrote {len(events)} rows to {path}")


if __name__ == "__main__":
    main()
