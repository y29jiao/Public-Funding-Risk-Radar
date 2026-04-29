"""Step 5: select candidates and run adverse-media search provider."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.media_search import generate_search_queries, run_media_search, select_media_candidates
from src.utils import configure_logging, read_csv_safe, write_csv


def _df(filename: str) -> pd.DataFrame:
    df = read_csv_safe(filename)
    return df if df is not None else pd.DataFrame()


def main() -> None:
    configure_logging()
    entity = _df("entity_funding_summary.csv")
    zombie = _df("zombie_signals.csv")
    concentration = _df("vendor_concentration.csv")
    market_shares = _df("vendor_market_shares.csv")
    candidates = select_media_candidates(entity, zombie, concentration, market_shares)
    queries = generate_search_queries(candidates)
    results = run_media_search(queries)
    print(f"Wrote {len(candidates)} rows to {write_csv(candidates, 'adverse_media_candidates.csv')}")
    print(f"Wrote {len(queries)} rows to {write_csv(queries, 'media_search_queries.csv')}")
    print(f"Wrote {len(results)} rows to {write_csv(results, 'media_search_results.csv')}")


if __name__ == "__main__":
    main()
