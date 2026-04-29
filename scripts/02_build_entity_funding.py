"""Step 2: build entity-level funding summary."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.funding_pipeline import build_entity_funding_summary
from src.utils import configure_logging, write_csv


def main() -> None:
    configure_logging()
    df = build_entity_funding_summary()
    path = write_csv(df, "entity_funding_summary.csv")
    print(f"Wrote {len(df)} rows to {path}")


if __name__ == "__main__":
    main()
