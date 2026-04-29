"""Step 6: build composite entity risk scores."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.risk_scoring import build_risk_scores
from src.utils import configure_logging, read_csv_safe, write_csv


def _df(filename: str) -> pd.DataFrame:
    df = read_csv_safe(filename)
    return df if df is not None else pd.DataFrame()


def main() -> None:
    configure_logging()
    scores = build_risk_scores(
        _df("entity_funding_summary.csv"),
        _df("zombie_signals.csv"),
        _df("vendor_market_shares.csv"),
        _df("vendor_concentration.csv"),
    )
    path = write_csv(scores, "entity_risk_scores.csv")
    print(f"Wrote {len(scores)} rows to {path}")


if __name__ == "__main__":
    main()
