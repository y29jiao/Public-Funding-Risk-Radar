"""Step 9: use LLM/mock entity disambiguation to reduce media false positives."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.entity_disambiguation import disambiguate_media_entities
from src.utils import configure_logging, read_csv_safe, write_csv


def main() -> None:
    configure_logging()
    media_results = read_csv_safe("media_search_results.csv")
    entities = read_csv_safe("entity_funding_summary.csv")
    if media_results is None:
        print("Missing outputs/media_search_results.csv; writing empty media_entity_matches.csv")
    if entities is None:
        print("Missing outputs/entity_funding_summary.csv; running without entity metadata")
    matches = disambiguate_media_entities(media_results, entities)
    path = write_csv(matches, "media_entity_matches.csv")
    print(f"Wrote {len(matches)} rows to {path}")
    print("Re-run scripts/08_classify_adverse_media.py to apply this false-positive filter.")


if __name__ == "__main__":
    main()
